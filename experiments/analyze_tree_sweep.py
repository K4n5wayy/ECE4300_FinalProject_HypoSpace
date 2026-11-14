"""Compare complete configurations in the deduplicated hard-question sweep."""
from collections import defaultdict
import json
from statistics import mean

from .direct_evaluation import ROOT
from .summarize_tree_sweep import summarize_case
from .tree_sweep_config import CONFIGURATIONS


def main():
    runs = {}
    for path in sorted((ROOT / "results").glob("mcts-tree_hard_quick_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data["tree_config"]
        key = tuple(config[field] for field in ("policy", "child_limit", "duplicate_reward"))
        if key in CONFIGURATIONS and data["completed"]:
            runs[key] = [summarize_case(case) for case in data["cases"]]
    missing = set(CONFIGURATIONS) - runs.keys()
    if missing:
        raise SystemExit(f"Waiting for {len(missing)} configurations")

    horizons = {}
    for index, case in enumerate(next(iter(runs.values()))):
        horizons[case["domain"]] = min(rows[index]["submissions"] for rows in runs.values())
    ranked = []
    for config, rows in runs.items():
        for row in rows:
            k = horizons[row["domain"]]
            row["shared_submissions"] = k
            row["distinct_at_shared_submissions"] = row["recovered_by_submission"][k - 1] if k else 0
        ranked.append({
            "config": config,
            "mean_coverage": mean(row["coverage"] for row in rows),
            "mean_coverage_at_shared_submissions": mean(
                row["distinct_at_shared_submissions"] / row["reference_count"] for row in rows
            ),
            "cases": rows,
        })
    ranked.sort(key=lambda row: row["mean_coverage"], reverse=True)

    def paired_effect(axis, left, right):
        deltas = defaultdict(list)
        for config, rows in runs.items():
            if config[axis] != left:
                continue
            target = (*config[:axis], right, *config[axis + 1:])
            if target in runs:
                for a, b in zip(rows, runs[target], strict=True):
                    deltas[a["domain"]].append(b["coverage"] - a["coverage"])
        return {domain: {"pairs": len(values), "mean_delta": mean(values)}
                for domain, values in deltas.items()}

    by_domain = {}
    for domain in horizons:
        entries = [(entry["config"], row) for entry in ranked
                   for row in entry["cases"] if row["domain"] == domain]
        entries.sort(key=lambda item: (-item[1]["coverage"], item[1]["submissions"]))
        config, best = entries[0]
        by_domain[domain] = {
            "best_config": config,
            "best_coverage": best["coverage"],
            "submissions": best["submissions"],
            "mean_final_coverage": mean(row["coverage"] for _, row in entries),
            "search_cap_stops": sum(row["stop_reason"] == "model_call_cap" for _, row in entries),
        }
    weakest = min(by_domain, key=lambda domain: by_domain[domain]["mean_final_coverage"])
    report = {
        "by_domain": by_domain,
        "next_experiment": {
            "focus": weakest,
            "starting_config": by_domain[weakest]["best_config"],
            "change": "Compare free-text steps with steps carrying verifier-checkable structural constraints.",
            "evaluation": "Hold model and submission rules fixed. Compare discovery by submission, "
                          "then validate on additional held-out hard questions.",
        },
        "matched_submission_horizons": horizons,
        "ranked_by_final_coverage": ranked,
        "seven_vs_five_children": paired_effect(1, 5, 7),
        "penalty_vs_no_penalty": paired_effect(2, 0, -1),
    }
    path = ROOT / "results" / "tree_sweep_analysis.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
