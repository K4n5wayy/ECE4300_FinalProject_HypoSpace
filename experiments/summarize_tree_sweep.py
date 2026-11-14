"""Summarize the hard three-question factorial experiment by submission."""
from collections import Counter
import json
from pathlib import Path

from .direct_evaluation import ROOT
from .tree_sweep_config import CONFIGURATIONS


def summarize_case(case):
    seen = set()
    curve = []
    for record in case["records"]:
        if record.get("submitted"):
            if record.get("reference_hit"):
                seen.add(json.dumps(record["hypothesis_key"]))
            curve.append(len(seen))
    return {
        "domain": case["domain"],
        "case_id": case["case_id"],
        "submissions": len(curve),
        "distinct_valid_answers": len(seen),
        "reference_count": case["reference_count"],
        "coverage": case["metrics"]["recovery"],
        "recovered_by_submission": curve,
        "stop_reason": case["stop_reason"],
        "tree_events": dict(Counter(record["tree_event"] for record in case["records"]
                                    if "tree_event" in record)),
        "requests": len(case["records"]),
        "reported_tokens": sum(record.get("total_tokens") or 0 for record in case["records"]),
    }


def main():
    runs = []
    for path in sorted((ROOT / "results").glob("mcts-tree_hard_quick_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["tree_config"]
        if (config["policy"], config["child_limit"], config["duplicate_reward"]) not in CONFIGURATIONS:
            continue
        runs.append({"file": path.name, **config,
                     "completed": result["completed"],
                     "cases": [summarize_case(case) for case in result["cases"]]})
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
