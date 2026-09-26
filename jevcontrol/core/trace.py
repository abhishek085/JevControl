"""Read the call log of an existing harness and work out which of its LLM calls are decisions.

This is the entry point for someone who has *not* instrumented anything: they export whatever their
app already logs (an OpenTelemetry dump, a LangSmith export, a few lines of their own logging), and
JevControl reconstructs the pipeline, then says which steps a System One model could answer instead.

The rule (stated by the user, and the one the classifier implements): a step can move to a decision
model only when its answer is **one of a fixed set, a level on a scale, or yes/no**, and the call needs
nothing beyond the standard decision prompt (state + question + options). Anything that writes text,
or needs bespoke instructions or reasoning, stays an LLM call.

Nothing here talks to a model. It is arithmetic and string analysis over the log the user already has.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- what we look for in an unknown log format -----------------------------------------------------
ID_KEYS = ("task_id", "trace_id", "session_id", "conversation_id", "run_id", "thread_id", "group", "example_id")
SITE_KEYS = ("site", "span", "span_name", "name", "step", "node", "operation", "event", "tag", "label")
PROMPT_KEYS = ("prompt", "messages", "input", "inputs", "request")
OUTPUT_KEYS = ("output", "completion", "response", "text", "content", "answer", "outputs", "generation")
IN_TOK_KEYS = ("prompt_tokens", "input_tokens", "tokens_in", "prompt_token_count")
OUT_TOK_KEYS = ("completion_tokens", "output_tokens", "tokens_out", "candidates_token_count")
LAT_KEYS = ("latency_ms", "duration_ms", "elapsed_ms", "took_ms", "latency", "duration", "elapsed")
TIME_KEYS = ("ts", "timestamp", "time", "start_time", "started_at", "created_at")
MODEL_KEYS = ("model", "model_name", "engine", "deployment")

BOOLS = {"true": "true", "false": "false", "yes": "true", "no": "false", "y": "true", "n": "false",
         "1": "true", "0": "false", "t": "true", "f": "false", "supported": "true", "unsupported": "false",
         "safe": "false", "unsafe": "true", "pass": "true", "fail": "false"}
# Prompt asks the model to think or explain: not a single-token decision, whatever the answer looks like.
REASONING = re.compile(r"step[- ]by[- ]step|chain of thought|think (?:it )?through|explain (?:your|why|the)"
                       r"|justif|reason(?:ing)? (?:for|why|first)|show your work|briefly (?:explain|say)"
                       r"|in your own words|elaborat", re.IGNORECASE)
# Sentences that only say how to format the reply: never the question being asked.
FORMAT_LINE = re.compile(r"(?i)\b(?:reply|respond|answer|return|output|format|print)\b[^.!?\n]{0,80}"
                         r"\b(?:json|only|exactly|single|one word|one letter|following format|schema)\b")
LEAD_IN = re.compile(r"(?i)^\s*(?:please\s+)?(?:decide|determine|assess|judge|check|evaluate|tell me|say)"
                     r"\s+(?:whether|if)\s+")
MAX_DECISION_TOKENS = 16   # a menu answer is one token; allow slack for `{"answer": "x"}`
HARD_GENERATION_TOKENS = 60  # above this there is no closed answer set to move, and the UI blocks overriding
MAX_OPTIONS = 60


def _flat(obj: Any, prefix: str = "", out: dict[str, Any] | None = None, depth: int = 0) -> dict[str, Any]:
    """Flatten nested dicts to dotted keys so an unknown schema can still be searched."""
    out = {} if out is None else out
    if depth > 4 or not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        out[key] = v
        if isinstance(v, dict):
            _flat(v, key + ".", out, depth + 1)
        elif isinstance(v, list) and v and isinstance(v[0], dict) and depth < 3:
            _flat(v[0], f"{key}.0.", out, depth + 1)
    return out


def _key(k: str) -> str:
    """Compare keys ignoring case and separators, so traceId / trace_id / TRACE-ID all match."""
    return re.sub(r"[^a-z0-9]", "", k.lower())


def _first(flat: dict[str, Any], names: tuple[str, ...], want: type | tuple[type, ...] = object) -> Any:
    """First value whose key is one of `names` (exactly, or as the last segment of a dotted path)."""
    norm = [(k, v, _key(k), _key(k.rsplit(".", 1)[-1])) for k, v in flat.items()]
    for n in names:
        target = _key(n)
        for _k, v, whole, last in norm:
            if (whole == target or last == target) and v is not None and isinstance(v, want):
                return v
    return None


def _messages_text(m: Any) -> str:
    if isinstance(m, str):
        return m
    if isinstance(m, list):
        parts = []
        for x in m:
            if isinstance(x, dict):
                c = x.get("content")
                if isinstance(c, list):  # content blocks
                    c = " ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                parts.append(f"{x.get('role', 'user')}: {c}")
            else:
                parts.append(str(x))
        return "\n".join(parts)
    if isinstance(m, dict):
        return _messages_text(m.get("messages") or m.get("content") or json.dumps(m, sort_keys=True))
    return "" if m is None else str(m)


def _as_messages(p: Any) -> list[dict[str, str]]:
    """Whatever was logged, as chat messages we can replay."""
    if isinstance(p, list) and p and isinstance(p[0], dict) and "content" in p[0]:
        out = []
        for x in p:
            c = x.get("content")
            if isinstance(c, list):
                c = " ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
            out.append({"role": str(x.get("role", "user")), "content": str(c)})
        return out
    if isinstance(p, dict) and isinstance(p.get("messages"), list):
        return _as_messages(p["messages"])
    return [{"role": "user", "content": _messages_text(p)}]


def _text_of(v: Any) -> str:
    """The reply as text, digging through the usual response envelopes."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if isinstance(v.get("choices"), list) and v["choices"]:
            return _text_of(v["choices"][0])
        if isinstance(v.get("candidates"), list) and v["candidates"]:  # Gemini-shaped
            return _text_of(v["candidates"][0])
        for k in ("content", "text", "answer", "output", "message", "completion", "generation", "parts"):
            if k in v:
                return _text_of(v[k])
        return json.dumps(v, sort_keys=True)
    if isinstance(v, list) and v:
        return _text_of(v[0])
    return "" if v is None else str(v)


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


class TraceError(ValueError):
    pass


@dataclass
class RawCall:
    task_id: str
    site: str
    order: int
    messages: list[dict[str, str]]
    prompt_text: str
    output: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    tokens_estimated: bool = False


def load_trace(path: str | Path, limit_tasks: int | None = None) -> list[RawCall]:
    """Parse a JSONL (or JSON array) call log. Unknown-but-reasonable schemas are handled by key sniffing."""
    p = Path(path).expanduser()
    if not p.exists():
        raise TraceError(f"file not found: {p}")
    raw = p.read_text(errors="replace").strip()
    if not raw:
        raise TraceError(f"{p.name} is empty")
    records: list[Any] = []
    if raw[0] == "[":
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as e:
            raise TraceError(f"{p.name}: not valid JSON ({e})") from e
    else:
        for i, line in enumerate(raw.splitlines(), 1):
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise TraceError(f"{p.name} line {i}: not valid JSON ({e}). One JSON object per line is expected.") from e
    calls: list[RawCall] = []
    skipped = 0
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            skipped += 1
            continue
        flat = _flat(rec)
        prompt = _first(flat, PROMPT_KEYS, (str, list, dict))
        output = _first(flat, OUTPUT_KEYS, (str, list, dict))
        if prompt is None or output is None:
            skipped += 1
            continue
        ptxt, otxt = _messages_text(prompt), _text_of(output)
        if not ptxt.strip() or not otxt.strip():
            skipped += 1
            continue
        lat = _first(flat, LAT_KEYS, (int, float)) or 0.0
        if lat and lat < 60 and not any(k.endswith("_ms") for k in flat):  # looks like seconds
            lat *= 1000.0
        pin = _first(flat, IN_TOK_KEYS, (int, float))
        pout = _first(flat, OUT_TOK_KEYS, (int, float))
        est = pin is None or pout is None
        tid = _first(flat, ID_KEYS, (str, int))
        site = _first(flat, SITE_KEYS, (str,))
        calls.append(RawCall(
            task_id=str(tid) if tid is not None else f"task-{i + 1}",
            site=str(site).strip() if site else "",
            order=int(_first(flat, TIME_KEYS, (int, float)) or i),
            messages=_as_messages(prompt), prompt_text=ptxt, output=otxt.strip(),
            prompt_tokens=int(pin) if pin is not None else est_tokens(ptxt),
            completion_tokens=int(pout) if pout is not None else est_tokens(otxt),
            latency_ms=float(lat), model=str(_first(flat, MODEL_KEYS, (str,)) or ""), tokens_estimated=est))
    if not calls:
        raise TraceError(
            f"{p.name}: found {len(records)} records but none had both a prompt and an output. "
            "Each line needs the prompt (prompt / messages / input) and what the model replied "
            "(output / completion / response / choices[0].message.content).")
    calls.sort(key=lambda c: (c.task_id, c.order))
    if limit_tasks:
        keep = list(dict.fromkeys(c.task_id for c in calls))[:limit_tasks]
        calls = [c for c in calls if c.task_id in set(keep)]
    _name_sites(calls)
    return calls


def _name_sites(calls: list[RawCall]) -> None:
    """Give every call a step name. If the log had none, group by prompt shape and number by position."""
    unnamed = [c for c in calls if not c.site]
    if not unnamed:
        return
    shapes: dict[str, list[RawCall]] = {}
    for c in unnamed:
        # a stable fingerprint of the prompt template: its long words, minus anything task-specific
        words = re.findall(r"[a-zA-Z]{4,}", c.prompt_text.lower())
        key = " ".join(sorted(set(words))[:40])
        shapes.setdefault(key, []).append(c)
    order = sorted(shapes.values(), key=lambda g: statistics.median(c.order for c in g))
    for i, group in enumerate(order, 1):
        for c in group:
            c.site = f"step{i}"


# --- the analysis ----------------------------------------------------------------------------------

def answer_value(text: str) -> str:
    """The answer a harness would actually use: unwrap a one-key JSON object, strip quotes and padding."""
    t = text.strip().strip("`").strip()
    m = re.search(r"\{[^{}]*\}", t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and len(obj) == 1:
                return str(list(obj.values())[0]).strip()
            if isinstance(obj, dict):
                for k in ("answer", "label", "tool", "choice", "value", "score", "decision", "category", "route"):
                    if k in obj:
                        return str(obj[k]).strip()
        except json.JSONDecodeError:
            pass
    return t.strip().strip('"\'').strip()


def _norm(v: str) -> str:
    return re.sub(r"\s+", " ", v).strip().lower()


def _num(v: str) -> float | None:
    try:
        return float(v)
    except ValueError:
        return None


def template_of(prompts: list[str]) -> tuple[str, str]:
    """Longest common prefix and suffix over a site's prompts: the fixed template around the varying state.

    The boundaries are snapped back to a token boundary. The raw longest common affixes can cut through a
    word - "PICK:true" and "PICK:false" share a trailing "e", which would leave the state ending "PICK:fals" -
    and a state truncated mid-token is both wrong and hard for a model to read.
    """
    if not prompts:
        return "", ""
    a = min(prompts, key=len)
    pre = 0
    while pre < len(a) and all(p[pre] == a[pre] for p in prompts):
        pre += 1
    suf = 0
    while suf < len(a) - pre and all(p[len(p) - 1 - suf] == a[len(a) - 1 - suf] for p in prompts):
        suf += 1

    if pre >= len(a) and suf == 0:  # nothing varies: the whole prompt is the template
        return a, ""

    def boundary(ch: str) -> bool:
        return not (ch.isalnum() or ch in "_-")

    while pre > 0 and not boundary(a[pre - 1]):   # the prefix must end at a boundary
        pre -= 1
    while suf > 0 and not boundary(a[len(a) - suf]):  # the suffix must start at one
        suf -= 1
    return a[:pre], a[len(a) - suf:] if suf else ""


def rubric_from_prompt(text: str) -> dict[str, str]:
    """Numeric rubric lines a scoring prompt usually carries, e.g. "0 = not relevant"."""
    out: dict[str, str] = {}
    for m in re.finditer(r"(?m)^\s*[-*\u2022]?\s*(\d{1,2})\s*(?:=|:|\)|\s-\s)\s*(.{2,120}?)\s*$", text):
        out.setdefault(m.group(1), m.group(2).strip())
    return out if len(out) >= 2 else {}


def options_from_prompt(text: str) -> dict[str, str]:
    """Option names (and definitions, when present) listed in the prompt itself."""
    out: dict[str, str] = {}
    block = re.search(r"(?:options?|choose (?:one )?(?:from|of)|one of|categor(?:y|ies)|labels?)\s*[:\n]"
                      r"((?:\s*[-*•]\s*.+\n?){2,})", text, re.IGNORECASE)
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"\s*[-*•]\s*([^:\n]{1,60}?)\s*(?::\s*(.+))?$", line)
            if m and m.group(1).strip():
                out[m.group(1).strip().strip('"\'`')] = (m.group(2) or "").strip()
    if not out:
        inline = re.search(r"(?:one of|answer with|reply with|respond with|either)\s*[:]?\s*"
                           r"([\w \-]+(?:\s*[|,/]\s*[\w \-]+){1,20})", text, re.IGNORECASE)
        if inline:
            for part in re.split(r"\s*[|,/]\s*", inline.group(1)):
                part = part.strip().strip('"\'`.')
                if part and len(part) < 40 and part.lower() not in ("or", "and"):
                    out[part] = ""
    if not out:  # "Classify the ticket as a or b", "Label it urgent, normal or low"
        listed = re.search(r"(?:classif\w*|label|categoris\w*|categoriz\w*|mark|tag|decide|answer|reply|respond)"
                           r"[^.\n]{0,40}?\b(?:as|with|into)\b\s+"
                           r"([\w-]{1,20}(?:\s*(?:,|/|\bor\b|\band\b)\s*[\w-]{1,20}){1,12})", text, re.IGNORECASE)
        if listed:
            for part in re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", listed.group(1), flags=re.IGNORECASE):
                part = part.strip().strip('"\'`.')
                if part and part.lower() not in ("an", "the", "either", "it", "this", "follows"):
                    out[part] = ""
            if len(out) < 2:
                out = {}
    return {k: v for k, v in out.items() if k}


@dataclass
class SiteAnalysis:
    site: str
    n: int
    kind: str                      # choice | score | noul | generation
    confidence: str                # high | medium | low
    reason: str                    # the evidence, in words, for the UI
    movable: bool                  # can it be a decision model call at all?
    overridable: bool              # may the user force it to be one?
    options: dict[str, str] = field(default_factory=dict)
    instructions: str = ""
    prefix: str = ""
    suffix: str = ""
    distinct: int = 0
    examples: list[str] = field(default_factory=list)
    med_prompt_tokens: int = 0
    med_out_tokens: int = 0
    med_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_out_tokens: int = 0
    calls_per_task: float = 0.0
    reasoning_prompt: bool = False


def _instructions(prefix: str, suffix: str, kind: str) -> str:
    """Recover the question the step actually asks, from the fixed part of the original prompt.

    The prefix usually opens with the instruction ("Decide whether...", "Rate how well...") and the suffix
    carries the output-format demand, which is not part of the question.
    """
    text = (prefix + "\n" + suffix).strip()
    text = re.sub(r"(?im)^\s*(?:system|user|assistant)\s*:\s*", "", text)
    pieces = [p.strip() for p in re.split(r"(?<=[.?!])\s+|\n+", text)]
    cands = [p for p in pieces if 12 <= len(p) <= 240 and not FORMAT_LINE.search(p)
             and not re.match(r"^\s*[-*\u2022]|^\s*\d{1,2}\s*[=:)]", p)  # option / rubric lines
             and not re.match(r"(?i)^(?:options?|scale|rubric|levels?|categories)\s*:?$", p)]
    q = next((p for p in cands if p.rstrip().endswith("?")), None)
    if q is None:
        lead = re.compile(r"(?i)^(?:please\s+)?(?:decide|determine|assess|judge|rate|score|classify|categoris|"
                          r"categoriz|pick|choose|select|route|assign|triage|flag|detect|identify|check|"
                          r"which|what|is |are |does |do |should |how )")
        q = next((p for p in cands if lead.match(p)), None)
    if q is None:
        verbs = re.compile(r"(?i)\b(which|whether|rate|classify|decide|pick|choose|select|score|route|triage|"
                           r"assign|relevant|urgent|spam|safe)\b")
        q = next((p for p in cands if verbs.search(p)), None)
    if q is None:  # fall back to the opening line: a prompt states its instruction before its data
        q = cands[0] if cands else ""
    q = re.sub(r"\s+", " ", q).strip(" -*:\u2022")
    if kind == "noul" and q:  # a Noul takes a claim, not a question
        q = LEAD_IN.sub("", q).strip()
        q = re.sub(r"(?i)^(?:it is true that|the following is true:)\s*", "", q).rstrip("?.").strip()
        if q:
            q = q[0].upper() + q[1:]
    if not q:
        q = {"noul": "The claim is true of the state.", "score": "Rate the state on the scale below.",
             "choice": "Which option applies to the state?"}[kind]
    return q[:240]


def analyze_site(site: str, calls: list[RawCall], n_tasks: int) -> SiteAnalysis:
    answers = [answer_value(c.output) for c in calls]
    norms = [_norm(a) for a in answers]
    counts = Counter(norms)
    n = len(calls)
    med_out = int(statistics.median(c.completion_tokens for c in calls))
    med_in = int(statistics.median(c.prompt_tokens for c in calls))
    med_lat = float(statistics.median(c.latency_ms for c in calls))
    prompts = [c.prompt_text for c in calls]
    pre, suf = template_of(prompts)
    prompt_opts = options_from_prompt(pre + "\n" + suf) or options_from_prompt(prompts[0])
    reasoning = sum(1 for p in prompts if REASONING.search(p)) > n / 2
    base = dict(site=site, n=n, distinct=len(counts), med_prompt_tokens=med_in, med_out_tokens=med_out,
                med_latency_ms=med_lat, total_prompt_tokens=sum(c.prompt_tokens for c in calls),
                total_out_tokens=sum(c.completion_tokens for c in calls), prefix=pre, suffix=suf,
                calls_per_task=n / max(n_tasks, 1), reasoning_prompt=reasoning,
                examples=[a[:120] for a in list(dict.fromkeys(answers))[:6]])

    def gen(reason: str, overridable: bool | None = None) -> SiteAnalysis:
        ov = (med_out <= HARD_GENERATION_TOKENS) if overridable is None else overridable
        return SiteAnalysis(kind="generation", confidence="high", reason=reason, movable=False,
                            overridable=ov, instructions=_instructions(pre, suf, "choice"),
                            options=prompt_opts, **base)

    if med_out > HARD_GENERATION_TOKENS:
        return gen(f"writes text: {med_out} output tokens per call (median), so there is no closed answer set")
    if reasoning:
        return gen("the prompt asks the model to explain or reason, which a single-pass decision cannot do")

    # yes/no -> Noul
    if len(counts) <= 3 and all(_norm(a) in BOOLS for a in answers):
        return SiteAnalysis(kind="noul", confidence="high", movable=True, overridable=True,
                            options={}, instructions=_instructions(pre, suf, "noul"),
                            reason=f"every one of {n} answers was yes/no ({med_out} output tokens)", **base)
    # a small ordered numeric set -> Score
    nums = [_num(a) for a in answers]
    if all(x is not None for x in nums) and len({*nums}) <= 12 and all(float(x).is_integer() for x in nums):  # type: ignore[arg-type]
        lo, hi = int(min(nums)), int(max(nums))  # type: ignore[type-var]
        rubric = rubric_from_prompt(pre + "\n" + suf) or rubric_from_prompt(prompts[0])
        levels = {str(v): "" for v in range(lo, hi + 1)}
        for src in (prompt_opts, rubric):
            for k, v in src.items():
                if k in levels and v:
                    levels[k] = v
        return SiteAnalysis(kind="score", confidence="high", movable=True, overridable=True, options=levels,
                            instructions=_instructions(pre, suf, "score"),
                            reason=f"all {n} answers were whole numbers in {lo}–{hi}", **base)
    # a small closed label set -> Choice
    opts = dict(prompt_opts)
    for a in dict.fromkeys(answers):  # labels actually seen, plus any the prompt listed
        if _norm(a) not in {_norm(k) for k in opts} and len(a) <= 60:
            opts[a] = ""
    enough_repeats = n <= 8 or len(counts) <= max(3, round(n * 0.6))
    if med_out <= MAX_DECISION_TOKENS and 2 <= len(opts) <= MAX_OPTIONS and enough_repeats:
        top = ", ".join(f"{k}" for k, _ in counts.most_common(3))
        conf = "medium" if (len(counts) < 2 or len(counts) > 12 or len(counts) >= n) else "high"
        return SiteAnalysis(kind="choice", confidence=conf, movable=True, overridable=True, options=opts,
                            instructions=_instructions(pre, suf, "choice"),
                            reason=(f"{n} calls produced {len(counts)} distinct short answers ({top}…), "
                                    f"{med_out} output tokens each" if len(counts) > 1 else
                                    f"all {n} logged answers were \"{top}\", and the prompt lists "
                                    f"{len(opts)} options, so the answer set is fixed"), **base)
    free_text = len(counts) > max(2, round(n * 0.6)) and med_out > MAX_DECISION_TOKENS
    if len(counts) == n and med_out <= MAX_DECISION_TOKENS:
        return gen(f"every one of {n} answers was different, so this is not a fixed answer set", True)
    return gen(f"{len(counts)} distinct answers over {n} calls, {med_out} output tokens: no closed answer set",
               not free_text)


@dataclass
class TraceReport:
    path: str
    n_calls: int
    n_tasks: int
    sites: list[SiteAnalysis]
    order: list[str]
    tokens_estimated: bool
    models: list[str]
    skipped: int = 0

    def movable(self) -> list[SiteAnalysis]:
        return [s for s in self.sites if s.movable]


def analyze(calls: list[RawCall], path: str = "") -> TraceReport:
    tasks = list(dict.fromkeys(c.task_id for c in calls))
    by_site: dict[str, list[RawCall]] = {}
    for c in calls:
        by_site.setdefault(c.site, []).append(c)
    pos: dict[str, list[int]] = {}  # position of each call within its own task, per site
    seen: dict[str, int] = {}
    for c in calls:
        i = seen.get(c.task_id, 0)
        seen[c.task_id] = i + 1
        pos.setdefault(c.site, []).append(i)
    order = sorted(by_site, key=lambda s: (statistics.median(pos[s]), s))
    sites = [analyze_site(s, by_site[s], len(tasks)) for s in order]
    return TraceReport(path=path, n_calls=len(calls), n_tasks=len(tasks), sites=sites, order=order,
                       tokens_estimated=any(c.tokens_estimated for c in calls),
                       models=sorted({c.model for c in calls if c.model}))


# --- what accepting a set of replacements would save (no model calls: arithmetic on the log) --------

def project(report: TraceReport, accept: dict[str, str], llm_in: float, llm_out: float,
            dec_in: float = 0.0, dec_out: float = 0.0) -> dict[str, Any]:
    """`accept` maps site -> primitive for the steps to move. Prices are $ per million tokens."""
    n = max(report.n_tasks, 1)
    cur_calls = sum(s.n for s in report.sites) / n
    cur_in = sum(s.total_prompt_tokens for s in report.sites) / n
    cur_out = sum(s.total_out_tokens for s in report.sites) / n
    new_calls = new_in = new_out = 0.0
    dec_calls = dec_tok = 0.0
    for s in report.sites:
        if s.site in accept:
            dec_calls += s.n / n
            dec_tok += s.total_prompt_tokens / n  # the state still has to be read, by the small model
        else:
            new_calls += s.n / n
            new_in += s.total_prompt_tokens / n
            new_out += s.total_out_tokens / n
    cost = (cur_in * llm_in + cur_out * llm_out) / 1e6
    new_cost = (new_in * llm_in + new_out * llm_out + dec_tok * dec_in + dec_calls * dec_out) / 1e6
    return {
        "n_tasks": report.n_tasks, "accepted": sorted(accept),
        "llm_calls": {"before": cur_calls, "after": new_calls},
        "decision_calls_after": dec_calls,
        "llm_prompt_tokens": {"before": cur_in, "after": new_in},
        "llm_output_tokens": {"before": cur_out, "after": new_out},
        "llm_token_reduction": 1 - (new_in + new_out) / max(cur_in + cur_out, 1e-9),
        "call_reduction": 1 - new_calls / max(cur_calls, 1e-9),
        "cost_per_1k": {"before": cost * 1000, "after": new_cost * 1000},
        "cost_reduction": (1 - new_cost / cost) if cost > 0 else None,
        "priced": cost > 0,
        "tokens_estimated": report.tokens_estimated,
        # Latency cannot be projected from a log: the decision model's speed is only known by running it.
        "llm_latency_removed_ms": sum(s.n * s.med_latency_ms for s in report.sites if s.site in accept) / n,
        "llm_latency_total_ms": sum(s.n * s.med_latency_ms for s in report.sites) / n,
    }


def report_json(report: TraceReport) -> dict[str, Any]:
    d = asdict(report)
    d["sites"] = [asdict(s) for s in report.sites]
    return d
