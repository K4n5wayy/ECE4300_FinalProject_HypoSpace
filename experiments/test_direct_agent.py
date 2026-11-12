"""Live smoke experiment, not an offline unit test. Run with python -m."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from agents import DirectAgent
from llm import ArkCodingPlanClient, LLMError
from .causal_case import build_case, ROOT


def main():
    parser = argparse.ArgumentParser(description="One causal scenario with independent model calls")
    parser.add_argument("--queries", type=int, default=5)
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("queries must be positive")
    benchmark, observations, references = build_case()
    prompt = benchmark.create_prompt(observations, [])
    agent = DirectAgent(ArkCodingPlanClient())
    reference_hashes = {g.get_hash() for g in references}
    unique, recovered, records = set(), set(), []
    valid_count = 0
    for index in range(args.queries):
        record = {"query": index + 1}
        try:
            response = agent.solve(prompt)
            record.update(asdict(response))
            graph = benchmark.parse_llm_response(response.text)
            valid = graph is not None and benchmark.validate_hypothesis(graph, observations)
            record.update(parsed=graph is not None, valid=valid)
            if graph:
                unique.add(graph.get_hash())
                if valid:
                    valid_count += 1
                    if graph.get_hash() in reference_hashes:
                        recovered.add(graph.get_hash())
            print(f"Query {index + 1}/{args.queries}: valid={valid}", flush=True)
        except LLMError as error:
            record["error"] = str(error)
            print(f"Query {index + 1}/{args.queries}: request failed", flush=True)
        records.append(record)
    metrics = {"validity": valid_count / args.queries,
               "uniqueness": len(unique) / args.queries,
               "recovery": len(recovered) / len(reference_hashes)}
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "agent": "DirectAgent", "history": False, "queries": args.queries,
              "note": "One constructed scenario; upstream prompt/parser/validator and metric definitions. No history unlike official runner.",
              "observations": observations, "reference_count": len(reference_hashes),
              "references": [g.to_dict() for g in references], "prompt": prompt,
              "max_tokens": agent.max_tokens,
              "upstream_sha256": hashlib.sha256((ROOT / "HypoSpace/_HypoSpace-main/causal/run_causal_benchmark.py").read_bytes()).hexdigest(),
              "metrics": metrics, "records": records}
    output = ROOT / "results" / ("direct_causal_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reference_count": len(reference_hashes), **metrics}))
    print(f"Saved: {output.name}")


if __name__ == "__main__":
    main()
