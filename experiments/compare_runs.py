"""Compare complete matched runs and emit paired case-level deltas."""
import argparse
import json
from pathlib import Path
from statistics import mean


METRICS = ("validity", "uniqueness", "recovery")


def read_run(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = {(row["domain"], row["case_id"]): row for row in data["cases"]}
    if not data["completed"] or len(cases) != data["expected_cases"]:
        raise ValueError(f"Incomplete run: {path}")
    calls = sum(len(row["records"]) for row in cases.values())
    if (calls > data["call_cap"] if data.get("early_stop")
            else calls != data["expected_requests"]):
        raise ValueError(f"Incorrect request count: {path}")
    return data, cases


def returned_models(cases):
    return {record["model"] for row in cases.values()
            for record in row["records"] if "model" in record}


def comparable_prompt(prompt):
    # Upstream renders this Boolean operator list from an unordered set.
    return prompt.replace("Use ONLY these operators: OR, AND", "Use ONLY these operators: AND, OR")


def recovery_trace(case):
    seen = set()
    trace = []
    for record in case["records"]:
        if record.get("reference_hit") and record.get("submitted", True):
            seen.add(json.dumps(record["hypothesis_key"], sort_keys=True))
        trace.append(len(seen))
    return trace


def compare(baseline, other):
    base, left = read_run(baseline)
    run, right = read_run(other)
    if (base.get("early_stop", False) != run.get("early_stop", False)
            or base["datasets"] != run["datasets"] or left.keys() != right.keys()):
        raise ValueError("Runs must use identical datasets and cases")
    base_models, run_models = returned_models(left), returned_models(right)
    if len(base_models) != 1 or base_models != run_models or base["max_tokens"] != run["max_tokens"]:
        raise ValueError("Runs must use one matching returned model and token limit")
    paired = []
    for (domain, case_id), case in left.items():
        trial = right[domain, case_id]
        same_budget = (case["submission_cap"] == trial["submission_cap"] if base.get("early_stop")
                       else len(case["records"]) == len(trial["records"]))
        if (not same_budget
                or comparable_prompt(case["prompt"]) != comparable_prompt(trial["prompt"])
                or case["reference_count"] != trial["reference_count"]):
            raise ValueError(f"Different case settings: {domain}/{case_id}")
        paired.append({"domain": domain, "case_id": case_id,
                       "baseline_recovered_by_call": recovery_trace(case),
                       "method_recovered_by_call": recovery_trace(trial),
                       "baseline_calls": len(case["records"]),
                       "method_calls": len(trial["records"]),
                       "baseline_submissions": case.get("submitted", len(case["records"])),
                       "method_submissions": trial.get("submitted", len(trial["records"])),
                       "baseline_seconds": case.get("seconds"),
                       "method_seconds": trial.get("seconds"),
                       **{metric: trial["metrics"][metric] - case["metrics"][metric]
                          for metric in METRICS}})

    totals = {}
    for group in base["summary"]:
        rows = [row for row in paired if group == "all" or row["domain"] == group]
        totals[group] = {"baseline": base["summary"][group], "method": run["summary"][group],
                         "delta": {metric: mean(row[metric] for row in rows) for metric in METRICS}}
    usage = {}
    for data, cases in ((base, left), (run, right)):
        records = [record for row in cases.values() for record in row["records"]]
        usage[data.get("method", data["agent"])] = {
            field: sum(record.get(field) or 0 for record in records)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    return {"baseline": baseline.name, "method": other.name, "model": next(iter(base_models)),
            "summary": totals, "reported_token_usage": usage, "paired_deltas": paired}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("methods", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps([compare(args.baseline, method) for method in args.methods], indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
