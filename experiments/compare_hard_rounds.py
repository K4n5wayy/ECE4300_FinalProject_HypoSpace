"""Compare the original and revised hard-quick methods at their actual costs."""
import json
from pathlib import Path
import sys

from .diagnose_hard_quick import analyze


def main():
    baseline_path, revised_path, output = map(Path, sys.argv[1:])
    base, revised = [json.loads(path.read_text(encoding="utf-8"))
                     for path in (baseline_path, revised_path)]
    assert base["completed"] and revised["completed"]
    assert base["datasets"] == revised["datasets"]
    cases = []
    for left, right in zip(base["cases"], revised["cases"], strict=True):
        fields = ("domain", "case_id", "reference_count", "submission_cap")
        assert all(left[field] == right[field] for field in fields)
        cases.append({
            **{field: left[field] for field in fields},
            "baseline": analyze(left["records"]),
            "revised": analyze(right["records"]),
            "coverage_delta": right["metrics"]["recovery"] - left["metrics"]["recovery"],
        })
    output.write_text(json.dumps({"baseline": baseline_path.name, "revised": revised_path.name,
                                  "summaries": [base["summary"], revised["summary"]],
                                  "cases": cases}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
