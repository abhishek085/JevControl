"""Tool layer: retrieval tools that both pipelines share.

A tool is a dict with a signature (``name``, ``description``, ``keywords``, ``args``)
and a ``run`` callable returning ``list[Document]``. Tools are deterministic and
shared across Pipeline A and B so the only difference between the two harnesses is
*how decisions are made around them*, not what data they see.

Bundled tools:
* ``vector_db_search``  — in-process cosine similarity over a tiny embedded corpus
* ``financial_report_fetch`` — cached synthetic 10-K/10-Q style filings
* ``news_scrape``       — cached synthetic news items (no live scraping in CI)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..drivers.llm import count_tokens


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return count_tokens(self.title + " " + self.text)

    def as_state_entry(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "url": self.url,
        }


@dataclass
class Tool:
    name: str
    description: str
    keywords: list[str]
    run: Callable[..., list[Document]]
    args_schema: dict[str, str] = field(default_factory=dict)

    def signature(self) -> dict[str, Any]:
        """What a router (LLM or Jev Choice) sees when deciding whether to call this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "keywords": list(self.keywords),
            "args": self.args_schema,
        }


# --- vector db search ---------------------------------------------------------------

# A tiny embedded corpus; enough to make ranking non-trivial.
_CORPUS: list[Document] = [
    Document("vdb-001", "vector-db", "ACME Corp Q2 earnings beat expectations",
             "ACME Corp reported Q2 revenue of 4.2 billion, beating consensus by 6%. Earnings per share was 2.31 versus 2.10 expected. The company raised full-year guidance.",
             "https://corp.example/acme-q2", {"ticker": "ACME", "period": "Q2"}),
    Document("vdb-002", "vector-db", "ACME Corp annual report 10-K summary",
             "The ACME Corp annual filing shows revenue of 15.8 billion for fiscal year, up 9% year over year, with operating margin of 21% and net income of 3.1 billion.",
             "https://corp.example/acme-10k", {"ticker": "ACME", "period": "FY"}),
    Document("vdb-003", "vector-db", "Tech market commentary: rate pressure",
             "Equity markets weighed in on rising rates as investors rotated out of growth names. Semiconductor indices fell 3% on the session.",
             "https://news.example/rates", {}),
    Document("vdb-004", "vector-db", "ACME supply chain: factory expansion",
             "ACME announced a 900 million expansion of its Austin manufacturing facility, expecting capacity to rise 18% by next fiscal year.",
             "https://corp.example/acme-factory", {"ticker": "ACME", "period": "guidance"}),
    Document("vdb-005", "vector-db", "Weather: front moving through",
             "A slow-moving front will bring scattered showers to the region over the next 48 hours with little impact on markets.",
             "https://news.example/weather", {}),
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embed(text: str) -> list[float]:
    """Deterministic bag-of-words vector (no model dependency, reproducible)."""
    vec: dict[str, int] = {}
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        vec[w] = vec.get(w, 0) + 1
    keys = sorted(vec)
    return [vec[k] for k in keys]


def _vdb_search(query: str, k: int = 5, source_filter: str | None = None) -> list[Document]:
    qv = _embed(query)
    # map query vector onto corpus keys
    scored = []
    for doc in _CORPUS:
        cv = _embed(doc.title + " " + doc.text)
        shared = {w for w in re.findall(r"[a-z0-9]+", query.lower())}
        overlap = sum(1 for w in shared if w in (doc.title + " " + doc.text).lower())
        scored.append((overlap, doc))
    scored.sort(key=lambda t: t[0], reverse=True)
    docs = [d for _, d in scored if (source_filter is None or d.source == source_filter)]
    return docs[:k]


def vector_db_search(query: str = "", k: int = 5, **_: Any) -> list[Document]:
    return _vdb_search(query=query, k=k)


# --- financial reports ---------------------------------------------------------------

_FINANCIAL_CORPUS: list[Document] = [
    Document("fin-001", "financial_report_fetch", "ACME Corp 10-Q (Q2)",
             "Form 10-Q. ACME Corp reports revenue 4.2B for the quarter, net income 980M, gross margin 41%. Management states inventory levels are normal.",
             "https://sec.example/acme-10q-q2", {"ticker": "ACME", "form": "10-Q"}),
    Document("fin-002", "financial_report_fetch", "ACME Corp 10-K (FY)",
             "Form 10-K. ACME Corp fiscal-year revenue 15.8B, net income 3.1B, R&D spend 1.4B. Risk factors cite concentration in two customers.",
             "https://sec.example/acme-10k", {"ticker": "ACME", "form": "10-K"}),
    Document("fin-003", "financial_report_fetch", "BEACON Inc 10-Q (Q2)",
             "Form 10-Q. BEACON Inc reports revenue 2.1B, a loss of 60M due to a one-time restructuring charge of 140M.",
             "https://sec.example/beacon-10q", {"ticker": "BEACON", "form": "10-Q"}),
]


def financial_report_fetch(ticker: str = "", form: str = "", **_: Any) -> list[Document]:
    docs = []
    for d in _FINANCIAL_CORPUS:
        if ticker and d.meta.get("ticker") != ticker.upper():
            continue
        if form and d.meta.get("form") != form:
            continue
        docs.append(d)
    return docs or list(_FINANCIAL_CORPUS[:2])


# --- news scrape (cached, deterministic) ---------------------------------------------

_NEWS_CORPUS: list[Document] = [
    Document("news-001", "news_scrape", "ACME tops estimates, lifts guidance",
             "ACME Corp beat Q2 estimates and raised full-year guidance, citing stronger-than-expected enterprise demand. Shares rose 4% after hours.",
             "https://news.example/acme-beat", {"ticker": "ACME"}),
    Document("news-002", "news_scrape", "Analyst note: ACME margin pressure",
             "An analyst flags possible margin pressure at ACME in H2 due to input-cost inflation, but sees guidance as achievable.",
             "https://news.example/acme-note", {"ticker": "ACME"}),
    Document("news-003", "news_scrape", "Markets: semis dip on rate fears",
             "Semiconductor stocks slipped on renewed rate worries. Analysts said the dip is a pullback, not a reversal.",
             "https://news.example/semis", {}),
]


def news_scrape(query: str = "", ticker: str = "", **_: Any) -> list[Document]:
    docs = []
    for d in _NEWS_CORPUS:
        if ticker and d.meta.get("ticker") != ticker.upper():
            continue
        if query and ticker == "" and query.lower() not in (d.title + " " + d.text).lower():
            continue
        docs.append(d)
    return docs or list(_NEWS_CORPUS)


# --- registry ------------------------------------------------------------------------


def default_tools() -> list[Tool]:
    return [
        Tool("vector_db_search",
             "Semantic search over a corpus of financial filings, news, and market notes. Returns top-k relevant passages.",
             ["search", "retrieve", "vector", "corpus", "passage", "document", "query"],
             vector_db_search, {"query": "str", "k": "int"}),
        Tool("financial_report_fetch",
             "Fetch formal corporate filings (10-K, 10-Q) for a ticker. Authoritative financial statements.",
             ["financial", "filing", "10-k", "10-q", "report", "ticker", "sec", "earnings", "revenue"],
             financial_report_fetch, {"ticker": "str", "form": "str"}),
        Tool("news_scrape",
             "Scrape recent news items for a ticker or topic. Fast but noisy; requires verification.",
             ["news", "article", "headline", "scrape", "reporting", "ticker", "story"],
             news_scrape, {"query": "str", "ticker": "str"}),
    ]


def tool_map(tools: list[Tool] | None = None) -> dict[str, Tool]:
    return {t.name: t for t in (tools or default_tools())}


def run_tool(tools: dict[str, Tool], name: str, **args: Any) -> list[Document]:
    return tools[name].run(**args)
