"""Diagnose coverage, duplicates, and request failures for hard quick runs."""
import argparse
from collections import Counter
import json
from math import ceil
from statistics import median
from pathlib import Path


def analyze(records):
    seen = set()
    outcomes = Counter()
    failures = Counter()
    curve = []
    for record in records:
        if "error" in record:
            outcomes["api_error"] += 1
            failures[record["error"]] += 1
        elif record.get("hypothesis_key") is None:
            outcomes["unparsed"] += 1
        elif not record.get("valid"):
            outcomes["invalid"] += 1
        elif not record.get("reference_hit"):
            outcomes["valid_outside_references"] += 1
        else:
            key = json.dumps(record["hypothesis_key"], sort_keys=True)
            outcomes["duplicate" if key in seen else "new_reference"] += 1
            seen.add(key)
        curve.append(len(seen))
    distributions = {}
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
        values = sorted(record[field] for record in records if record.get(field) is not None)
        if values:
            distributions[field] = {"count": len(values), "min": values[0], "median": median(values),
                                    "p90": values[ceil(len(values) * 0.9) - 1], "max": values[-1]}
    return {
        "token_distributions": distributions,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records if r.get("finish_reason"))),
        "calls": len(records),
        "submissions": sum(bool(record.get("submitted")) for record in records),
        "reported_tokens": sum(record.get("total_tokens") or 0 for record in records),
        "outcomes": dict(outcomes),
        "api_errors": dict(failures),
        "recovered_by_call": curve,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        cases = run["cases"]
        print(json.dumps({
            "file": path.name,
            "method": run["method"],
            "completed": run["completed"],
            "cases": [{"domain": case["domain"], "case_id": case["case_id"],
                       "reference_count": case["reference_count"],
                       "coverage": case["metrics"]["recovery"], **analyze(case["records"])}
                      for case in cases],
            "interrupted": analyze(run["interrupted_case"]["records"])
            if run.get("interrupted_case") else None,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
