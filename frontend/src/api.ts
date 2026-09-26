export type Endpoint = {
  name: string; base_url: string; model: string; api_key: string;
  price_in_per_m: number; price_out_per_m: number; extra_body: Record<string, unknown>; timeout_s: number;
};
export type ArmKind = "baseline" | "menu" | "hybrid";
export type Arm = { id: string; label: string; kind: ArmKind; decider: Endpoint | null; tau: number; temperature: number; temperatures?: Record<string, number>; tau_by_site?: Record<string, number> };
export type HarnessRef = { demo?: string | null; path?: string | null; tasks?: string | null };
export type ExperimentConfig = {
  name: string; harness: HarnessRef; llm: Endpoint; arms: Arm[]; n_tasks: number | null; concurrency: number;
  seed: number; bootstrap: number; margin: number; scorer: "harness" | "exact" | "contains"; expected_field: string;
};
export type HarnessInfo = {
  id: string; name: string; description: string; sites: Record<string, string>; n_tasks: number; sample_task: Record<string, unknown>;
  has_score: boolean; has_truth: boolean; tools: string[]; path: string; tasks_path: string; kinds: Record<string, number>; demo?: string;
};
export type ProbeResult = {
  ok: boolean; models: string[]; chat_ok: boolean; logprobs_ok: boolean; latency_ms: number; error: string; model: string;
  label_mass?: number; sample_probs?: number[]; readout_ms?: number;
};
export type ArmSummary = {
  id: string; label: string; kind: ArmKind; tau: number; decider: string | null; n: number; errors: number; accuracy: number;
  e2e_ms: { mean: number; p50: number; p95: number }; llm_calls: number; llm_generate_calls: number; llm_decide_calls: number;
  llm_prompt_tokens: number; llm_completion_tokens: number; decider_calls: number; decider_ms: number; cost_per_1k: number;
  decisions_per_task: number; offload_rate: number; escalation_rate: number; parse_fail_rate: number;
  decision_accuracy: number | null; decision_truth_n: number; label_mass: number | null;
};
export type Paired = {
  n: number; delta_acc: number; ci: [number, number]; wins: number; losses: number; ties: number; mcnemar_p: number;
  speedup: number; speedup_ci: [number, number]; token_reduction: number; calls_reduction: number; cost_reduction: number | null;
  verdict: "safe" | "unclear" | "worse"; margin: number; leans_worse: boolean; tasks_needed: number | null; flips: { task_id: string; base: number; cand: number }[];
};
export type SiteRow = {
  site: string; kind: string; n: number; offload?: number; mean_confidence?: number; latency_ms: number; base_latency_ms?: number;
  agreement?: number | null; agreement_n?: number; truth_acc: number | null; base_truth_acc?: number | null; truth_n?: number;
  parse_fail_rate?: number;
};
export type SweepPoint = { tau: number; offload: number; agreement: number | null; n: number; hybrid_acc?: number; base_acc?: number; truth_n?: number };
export type Sweep = { points: SweepPoint[]; recommended: { tau: number; offload: number; agreement: number | null; basis: string } | null; n: number };
export type Summary = {
  arms: ArmSummary[]; paired: Record<string, Paired>; sites: Record<string, SiteRow[]>;
  callmap?: Record<string, CallMapRow[]>;
  sweeps: Record<string, { overall: Sweep; sites: Record<string, Sweep> }>; baseline: string | null; margin: number;
  wall_s?: number; cancelled?: boolean;
};
export type RunMeta = { id: string; name: string; status: string; created: number; error: string; arms: string[]; n_tasks: number | null };
export type RunDetail = RunMeta & { config: ExperimentConfig; progress: Record<string, { done: number; n: number }>; summary: Summary | null };
export type Decision = {
  site: string; kind: string; selected: string; confidence: number | null; probabilities: Record<string, number>; value: number | null;
  source: string; escalated: boolean; menu_confidence: number | null; menu_selected: string | null; latency_ms: number; key: string | null;
  parsed: boolean; truth: string | null; correct: boolean | null;
};
export type Row = {
  arm: string; task_id: string; score: number; error: string | null; output: unknown; e2e_ms: number; llm_calls: number;
  llm_prompt_tokens: number; llm_completion_tokens: number; decisions: Decision[];
  calls: { role: string; endpoint: string; site: string | null; latency_ms: number; prompt_tokens: number; completion_tokens: number; on_main_llm: boolean }[];
  tools: { name: string; args: Record<string, unknown>; cached: boolean; latency_ms: number }[];
};
export type TaskLine = { id: string; preview: string; kind: string; arms: Record<string, { score: number; e2e_ms: number; llm_calls: number; error: boolean }> };
export type LocalModel = { id: string; path: string; source: string; size_gb: number; arch: string; quant: string | null; decision_model: boolean };
export type Server = { name: string; status: string; running: boolean; port: number; model: string; served_name: string; ready: boolean; base_url: string; managed?: boolean };
export type Job = { id: string; repo_id: string; status: string; bytes_done: number; bytes_total: number; error: string };
export type ModelsInfo = {
  local: LocalModel[]; servers: Server[]; jobs: Job[]; catalog: { repo_id: string; role: string; title: string; note: string }[];
};

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : undefined, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* keep default */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}
export const api = {
  get: <T,>(p: string) => req<T>("GET", p),
  post: <T,>(p: string, b?: unknown) => req<T>("POST", p, b ?? {}),
  del: <T,>(p: string) => req<T>("DELETE", p),
};

export const blankEndpoint = (over: Partial<Endpoint> = {}): Endpoint => ({
  name: "", base_url: "http://localhost:8000/v1", model: "", api_key: "EMPTY", price_in_per_m: 0, price_out_per_m: 0,
  extra_body: { chat_template_kwargs: { enable_thinking: false } }, timeout_s: 120, ...over,
});

// ---- importing an existing harness's call log ------------------------------------------------------
export type SiteAnalysis = {
  site: string; n: number; kind: "choice" | "score" | "noul" | "generation";
  confidence: "high" | "medium" | "low"; reason: string; movable: boolean; overridable: boolean;
  options: Record<string, string>; instructions: string; prefix: string; suffix: string; distinct: number;
  examples: string[]; med_prompt_tokens: number; med_out_tokens: number; med_latency_ms: number;
  total_prompt_tokens: number; total_out_tokens: number; calls_per_task: number; reasoning_prompt: boolean;
};
export type TraceReport = {
  path: string; n_calls: number; n_tasks: number; sites: SiteAnalysis[]; order: string[];
  tokens_estimated: boolean; models: string[];
};
export type Projection = {
  n_tasks: number; accepted: string[]; llm_calls: { before: number; after: number };
  decision_calls_after: number; llm_prompt_tokens: { before: number; after: number };
  llm_output_tokens: { before: number; after: number }; llm_token_reduction: number; call_reduction: number;
  cost_per_1k: { before: number; after: number }; cost_reduction: number | null; priced: boolean;
  tokens_estimated: boolean; llm_latency_removed_ms: number; llm_latency_total_ms: number;
};
export type BuiltHarness = {
  dir: string; harness: string; tasks: string; n_tasks: number; moved: string[];
  sites: Record<string, { kind: string; options: Record<string, string>; instructions: string }>;
  projection: Projection; name: string;
};
export type ExampleTrace = { path: string; name: string; size_kb: number };
export type CallMapRow = {
  site: string; kind: string; role: string; answered_by: "llm" | "decision model" | "mixed";
  calls_per_task: number; calls: number; prompt_tokens_per_task: number; out_tokens_per_task: number;
  med_latency_ms: number; pos: number;
};

export const PRIMITIVES = {
  choice: { label: "Choice", hint: "pick one of a fixed set of options" },
  score: { label: "Score", hint: "a level on an ordered scale" },
  noul: { label: "Noul", hint: "a calibrated probability that a claim is true" },
  generation: { label: "LLM", hint: "writes text: stays on your main model" },
} as const;
