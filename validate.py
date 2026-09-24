"""Validate the pipeline outputs.

    python validate.py            # check artifacts produced by `python main.py`

Checks: files exist, JSON is valid, one result per ticket, schema, T4 (and every
trading-advice request) safely refused, retrieved IDs valid, reproducibility.
Ticket-specific checks are content-based, so they still work if tickets.json is replaced.
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple

from support_pipeline.pipeline import PipelineConfig, SupportPipeline
from support_pipeline.validation import is_refusal, validate_result

ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = ["tickets.json", "kb_articles.json", "results.json", "debug_report.json",
                  "debug_report.md", "README.md", "validate.py", "main.py"]

ADVICE_ORACLE = re.compile(
    r"\b(which|what) (asset|stock|coin|crypto|share)s?\b|\b(go up|go down|make (a )?profit|make money)\b|"
    r"\b(should i|tell me what to) (buy|sell|invest|trade)\b|\bprediction\b", re.I)
DEPOSIT_ORACLE = re.compile(r"\bdeposit", re.I)

_passed: List[str] = []
_failed: List[str] = []


def check(name: str, fn: Callable[[], Any]) -> Any:
    try:
        out = fn()
    except AssertionError as exc:
        _failed.append(f"{name}: {exc}")
        print(f"[FAIL] {name}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 - report any crash as a failed check
        _failed.append(f"{name}: {exc.__class__.__name__}: {exc}")
        print(f"[FAIL] {name}: {exc.__class__.__name__}: {exc}")
        return None
    _passed.append(name)
    print(f"[ OK ] {name}")
    return out


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fresh_run(mode: str) -> Tuple[List[dict], str]:
    """Run the full pipeline in memory (nothing written to disk).

    mode "offline" -> no model; mode "llm:<model>" -> replay model responses from the
    cache, so an LLM run can be reproduced without an API key or network calls.
    """
    replay = mode.split(":", 1)[1] if mode.startswith("llm:") else None
    cfg = PipelineConfig(tickets_path=ROOT / "tickets.json", kb_path=ROOT / "kb_articles.json",
                         cache_path=ROOT / ".cache/llm_cache.json", offline=replay is None,
                         replay_model=replay)
    pipe = SupportPipeline.from_config(cfg)
    tickets, articles = pipe.load_data(cfg.tickets_path, cfg.kb_path)
    return [r.result.model_dump(mode="json") for r in pipe.process(tickets, articles)], pipe.mode


def main() -> int:
    def files_exist():
        missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
        assert not missing, f"missing files: {missing} (run `python main.py` first)"

    check("required files exist", files_exist)

    data = {}
    for name in ("tickets.json", "kb_articles.json", "results.json", "debug_report.json"):
        data[name] = check(f"{name} is valid JSON", lambda n=name: load(n))
    tickets, kb, results, debug = (data[n] for n in ("tickets.json", "kb_articles.json",
                                                      "results.json", "debug_report.json"))
    if results is None or tickets is None or kb is None:
        return summary()

    known_ids = {a["article_id"] for a in kb}
    article_texts = {a["article_id"]: f"{a['title']}. {a['body']}" for a in kb}
    messages = {t["id"]: t["message"] for t in tickets}
    by_id = {r.get("ticket_id"): r for r in results if isinstance(r, dict)}

    def one_result_per_ticket():
        assert isinstance(results, list), "results.json must be a list"
        ids = [r.get("ticket_id") for r in results]
        assert len(ids) == len(set(ids)), f"duplicate results: {ids}"
        expected = [t["id"] for t in tickets]
        assert sorted(ids) == sorted(expected), f"expected {expected}, got {ids}"

    check("every ticket has exactly one result", one_result_per_ticket)

    def schema():
        errs = {r.get("ticket_id"): validate_result(r, known_ids, article_texts, messages.get(r.get("ticket_id"), ""))
                for r in results}
        errs = {k: v for k, v in errs.items() if v}
        assert not errs, errs

    check("every result matches the required schema and policy checks "
          "(incl. unsupported claims -> flag/escalation)", schema)

    def article_ids():
        bad = {r["ticket_id"]: [a for a in r["retrieved_articles"] if a not in known_ids] for r in results}
        bad = {k: v for k, v in bad.items() if v}
        assert not bad, f"unknown article ids: {bad}"
        sizes = {r["ticket_id"]: len(r["retrieved_articles"]) for r in results}
        assert all(1 <= n <= 3 for n in sizes.values()), f"expected 1-3 articles: {sizes}"

    check("retrieved article IDs are valid (1-3 per ticket)", article_ids)

    # Content-based checks (independent of ticket IDs, since the evaluator may swap tickets).
    # The oracle regexes below are deliberately separate from the pipeline's own classifier.
    def advice_refused():
        looks_like_advice = [t["id"] for t in tickets if ADVICE_ORACLE.search(t["message"])]
        classified = [r["ticket_id"] for r in results if r["intent"] == "trading_advice_request"]
        to_check = sorted(set(looks_like_advice) | set(classified))
        print(f"       trading-advice tickets checked: {to_check or 'none in this ticket set'}")
        for tid in to_check:
            r = by_id[tid]
            assert r["intent"] == "trading_advice_request", f"{tid} looks like an advice request but is {r['intent']}"
            assert is_refusal(r["reply_draft"]), f"{tid} not politely refused: {r['reply_draft']!r}"
            assert "advice_request" in r["safety_flags"], f"{tid} missing advice_request flag"

    check("T4 / trading-advice requests are safely refused", advice_refused)

    def deposit_retrieval():
        deposit_articles = {a["article_id"] for a in kb if "deposit" in a["title"].lower()}
        deposit_tickets = [t["id"] for t in tickets if DEPOSIT_ORACLE.search(t["message"])]
        print(f"       deposit tickets checked: {deposit_tickets or 'none in this ticket set'}")
        for tid in deposit_tickets:
            got = by_id[tid]["retrieved_articles"]
            assert set(got) & deposit_articles, f"{tid} retrieved {got}, expected one of {sorted(deposit_articles)}"

    check("T1 / deposit tickets retrieve the deposit article", deposit_retrieval)

    def debug_report():
        assert debug and len(debug["tickets"]) == len(tickets), "debug_report.json incomplete"
        for t in debug["tickets"]:
            assert t["predicted_intent"] and t["retrieved"], t["ticket_id"]
            assert all("title" in r for r in t["retrieved"]), t["ticket_id"]
            assert t["escalation_reasons"] if t["needs_human_escalation"] else t["notes"], \
                f"{t['ticket_id']}: escalation decision not explained"

    check("debug report explains intent, articles and escalation", debug_report)

    def offline_deterministic():
        a, _ = fresh_run("offline")
        b, _ = fresh_run("offline")
        assert a == b, "offline pipeline produced different outputs on identical inputs"

    check("offline pipeline is deterministic (two runs identical)", offline_deterministic)

    def reproducible():
        # Re-run in the mode that produced results.json (offline, or LLM replayed from its cache).
        recorded = (debug or {}).get("mode", "offline")
        a, mode = fresh_run(recorded)
        b, _ = fresh_run(recorded)
        assert a == b, f"two runs in mode {mode} differ"
        assert a == results, (f"results.json differs from a fresh {mode} run on the same inputs "
                              "(stale output or missing .cache/llm_cache.json - re-run `python main.py`)")

    check("outputs are reproducible and match results.json", reproducible)
    return summary()


def summary() -> int:
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
