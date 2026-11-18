"""Recalculate summary excluding request-error-skipped cases and list cases to rerun.

The long-running workers keep the old summary in memory; this rescore the coverage
metrics over normally-finished cases only and prints the skipped cases that should
be rerun later.
"""
import argparse
import json
from pathlib import Path

from .direct_evaluation import summarize


def rescore(run):
    summary = summarize(run["cases"])
    skipped = sorted({
        (case["domain"], case["case_id"]) for case in run["cases"]
        if case.get("stop_reason") == "request_errors"
    })
    return {
        "file": run.get("_file", ""),
        "method": run.get("method"),
        "completed": run.get("completed"),
        "summary": summary,
        "skipped_cases_to_rerun": [{"domain": domain, "case_id": case_id}
                                   for domain, case_id in skipped],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Result JSON files to rescore")
    args = parser.parse_args()
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        run["_file"] = path.name
        print(json.dumps(rescore(run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
