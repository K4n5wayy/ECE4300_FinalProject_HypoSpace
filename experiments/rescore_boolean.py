"""Rescore saved Boolean candidates without changing search records."""
import argparse
import json
from pathlib import Path

from llm import ArkCodingPlanClient, LLMError
from .boolean_scoring import assist_boolean_score
from .direct_evaluation import ROOT, load_domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--llm-assist", action="store_true")
    args = parser.parse_args()
    benchmark, prepare, score = load_domain("boolean", ROOT / "datasets/hard/boolean.json")
    cases = {case["observation_set_id"]: case for case in benchmark.all_observation_sets}
    client = ArkCodingPlanClient() if args.llm_assist else None
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        for case in run["cases"]:
            if case["domain"] != "boolean":
                continue
            observations, references = prepare(cases[case["case_id"]])
            direct, assisted = set(), set()
            assist_errors = 0
            for record in case["records"]:
                answer = record.get("answer", record.get("text", ""))
                if not answer:
                    continue
                key, valid = score(answer, observations)
                if valid and key in references:
                    direct.add(key)
                elif key is None and client:
                    try:
                        key, valid = assist_boolean_score(answer, observations, score, client)
                    except LLMError:
                        assist_errors += 1
                        continue
                    if valid and key in references:
                        assisted.add(key)
            print(json.dumps({
                "run": path.name, "case": case["case_id"],
                "original_hits": round(case["metrics"]["recovery"] * len(references)),
                "parsed_hits": len(direct), "additional_assisted_hits": len(assisted - direct),
                "reference_count": len(references), "assist_errors": assist_errors,
            }))


if __name__ == "__main__":
    main()
