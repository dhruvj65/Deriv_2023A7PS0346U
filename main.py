"""Run the support-ticket pipeline.

    python main.py              # uses Gemini if GEMINI_API_KEY is set, else offline
    python main.py --offline    # force the deterministic, key-free path
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from support_pipeline import PipelineConfig, run
from support_pipeline.pipeline import PipelineError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickets", default="tickets.json")
    parser.add_argument("--kb", default="kb_articles.json")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--offline", action="store_true", help="do not call any LLM")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = PipelineConfig(tickets_path=Path(args.tickets), kb_path=Path(args.kb),
                            results_path=Path(args.out), offline=args.offline)
    try:
        runs = run(config)
    except PipelineError as exc:
        logging.error("pipeline failed: %s", exc)
        return 1

    for r in runs:
        res = r.result
        print(f"{res.ticket_id}: intent={res.intent.value:<24} articles={','.join(res.retrieved_articles):<8} "
              f"conf={res.confidence:.2f} escalate={str(res.needs_human_escalation):<5} "
              f"flags={[f.value for f in res.safety_flags]}")
    mode = json.loads(config.debug_json_path.read_text(encoding="utf-8"))["mode"]
    print(f"\nWrote {config.results_path}, {config.debug_json_path}, {config.debug_md_path} (mode: {mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
