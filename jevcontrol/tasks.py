"""Benchmark tasks for the multi-source retrieval & fact-checking pipeline."""

from __future__ import annotations

from typing import Any

TASKS: list[dict[str, Any]] = [
    {
        "id": "acme-q2-research",
        "text": (
            "Retrieve recent corporate financial filings, news, and market data for ACME Corp, "
            "verify contradictory facts (revenue figures, guidance changes), filter out noise, "
            "and generate a concise summary of the quarter's performance."
        ),
        "params": {"ticker": "ACME", "period": "Q2"},
    },
    {
        "id": "acme-annual-review",
        "text": (
            "Retrieve ACME Corp's annual financial report and recent news coverage, check whether "
            "reported revenue and margin claims match the filing, filter noise, and summarize "
            "the year with a risk note on customer concentration."
        ),
        "params": {"ticker": "ACME", "period": "FY"},
    },
    {
        "id": "beacon-earnings-check",
        "text": (
            "Retrieve BEACON Inc's latest quarterly filing and news articles, verify the claim that "
            "the company beat expectations against the reported revenue and loss figures, filter "
            "noise, and summarize what drove the quarter."
        ),
        "params": {"ticker": "BEACON", "period": "Q2"},
    },
    {
        "id": "semis-market-brief",
        "text": (
            "Retrieve recent news and market data on the semiconductor sector, verify contradictory "
            "headlines about rate pressure, filter noise, and write a short briefing on sector sentiment."
        ),
        "params": {"sector": "semiconductors"},
    },
]
