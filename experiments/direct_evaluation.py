"""Shared HypoSpace evaluation for the quick test and full direct baseline."""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
import sys
from time import sleep

from agents import DirectAgent
from agents.reasoning_tree import ReasoningTreeAgent
from agents.search import MCTSAgent, MemoryAgent, StructuredMemoryAgent
from agents.coverage_search import CoverageMCTSAgent
from llm import ArkCodingPlanClient, LLMError

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "HypoSpace" / "_HypoSpace-main"


def load_domain(domain, path):
    for name in ("causal", "3d", "boolean"):
        directory = str(UPSTREAM / name)
        if directory not in sys.path:
            sys.path.append(directory)

    if domain == "causal":
        from run_causal_benchmark import CausalBenchmarkEnhanced, CausalGraph

        benchmark = CausalBenchmarkEnhanced(str(path))

        def prepare(case):
            observations = case["observations"]
            keys = {CausalGraph.from_dict(graph).get_hash()
                    for graph in case["ground_truth_graphs"]}
            return observations, keys

        def score(text, observations):
            graph = benchmark.parse_llm_response(text)
            if graph is None:
                return None, False
            return graph.get_hash(), benchmark.validate_hypothesis(graph, observations)

    elif domain == "3d":
        from run_3d_benchmark import Benchmark3D, Structure3D

        benchmark = Benchmark3D(str(path))

        def prepare(case):
            keys = set()
            size = benchmark.metadata["grid_size"]
            for structure in case["ground_truth_structures"]:
                layers = [[list(map(int, layer[start:start + size]))
                           for start in range(0, len(layer), size)]
                          for layer in structure["layers"]]
                keys.add(Structure3D(layers).get_hash())
            return case["observation"], keys

        def score(text, observations):
            structure = benchmark.parse_llm_response(text)
            if structure is None:
                return None, False
            return (structure.get_hash(),
                    len(structure.layers) <= benchmark.metadata["max_height"]
                    and benchmark.validate_structure_matches_observations(structure, observations))

    else:
        from boolean_benchmark import BooleanBenchmarkRefined, BooleanObservation
        from boolean_dataset import BooleanDiscoveryGame

        benchmark = BooleanBenchmarkRefined(str(path))
        # Use the generator's hypothesis space, including its primitive NOR depth.
        expressions = {
            expression.mechanistic_key(**benchmark.mechanistic_opts): expression
            for expression in BooleanDiscoveryGame.generate_all_expressions(
                benchmark.variables, benchmark.operators, benchmark.max_depth,
                benchmark.mechanistic_opts,
            )
        }

        def prepare(case):
            observations = [BooleanObservation(obs["inputs"], obs["output"])
                            for obs in case["observations"]]
            keys = {ast.literal_eval(expr["mechanistic_key"])
                    for expr in case["ground_truth_expressions"]}
            return observations, keys

        tokens = "|".join(re.escape(token)
                          for token in sorted(set(benchmark.variables) | benchmark.operators))
        allowed_expression = re.compile(rf"(?:\s+|[(),]|\b(?:{tokens})\b)+")

        def score(text, observations):
            expression = benchmark.parse_llm_response(text)
            if expression is None or not allowed_expression.fullmatch(expression):
                return None, False
            key = benchmark.get_expression_mechanistic_key(expression)
            if key is None:
                return None, False
            representative = expressions.get(key)
            valid = representative is not None and all(
                representative.evaluate(observation.inputs) == observation.output
                for observation in observations
            )
            return key, valid

    return benchmark, prepare, score


def summarize(cases):
    summary = {}
    for group in ("all", "causal", "3d", "boolean"):
        rows = [case for case in cases if group == "all" or case["domain"] == group]
        if rows:
            summary[group] = {
                "cases": len(rows),
                "requests": sum(len(row["records"]) for row in rows),
                "submissions": sum(row.get("submitted", len(row["records"])) for row in rows),
                "total_tokens": sum(record.get("total_tokens") or 0 for row in rows for record in row["records"]),
                "request_errors": sum("error" in record for row in rows for record in row["records"]),
                **{metric: mean(row["metrics"][metric] for row in rows)
                   for metric in ("validity", "uniqueness", "recovery")},
            }
    return summary


def wait_if_paused(pause_file):
    if not pause_file.exists():
        return
    print(f"Paused. Remove {pause_file} to continue.", flush=True)
    while pause_file.exists():
        sleep(1)
    print("Resumed.", flush=True)


def main(*, quick_cases=None, medium=False, hard=False):
    quick = quick_cases is not None
    early_stop = medium or hard
    dataset_root = ROOT / "datasets"
    if hard:
        dataset_root /= "hard"
    medium_cases = json.loads((ROOT / "datasets/medium_cases.json").read_text()) if medium else None
    parser = argparse.ArgumentParser(description="Quick test" if quick else "Full HypoSpace baseline")
    parser.add_argument("--queries", type=int, default=None,
                        help="Fixed calls per case. Default: one call per distinct reference hypothesis.")
    parser.add_argument("--method", choices=("direct", "memory", "memory-structured", "mcts", "mcts-breadth", "mcts-recent", "mcts-llm", "mcts-tree"), default="direct")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted direct, UCT or reasoning-tree run")
    token_limit = parser.add_mutually_exclusive_group()
    token_limit.add_argument("--max-tokens", type=int, default=2048)
    token_limit.add_argument("--provider-token-limit", dest="max_tokens", action="store_const", const=None)
    parser.add_argument("--tree-policy", choices=("uct", "breadth", "recent", "llm"), default="uct")
    parser.add_argument("--exploration-list", choices=("on", "off"),
                        help="Compare LLM node selection with or without round-based deferral")
    parser.add_argument("--child-limit", type=int, choices=(5, 7), default=5)
    parser.add_argument("--duplicate-reward", type=int, choices=(0, -1), default=0)
    parser.add_argument("--model")
    parser.add_argument("--pause-file", type=Path, default=ROOT / "results" / "PAUSE",
                        help="Pause between requests while this file exists")
    parser.add_argument("--plan", action="store_true", help="Show workload without model calls")
    args = parser.parse_args()
    if args.queries is not None and args.queries < 1:
        parser.error("--queries must be positive")

    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.exploration_list and (args.method != "mcts-tree" or args.tree_policy != "llm"):
        parser.error("--exploration-list requires --method mcts-tree --tree-policy llm")
    tree = args.method.startswith("mcts")
    jobs = []
    datasets = {}
    for domain in ("causal", "3d", "boolean"):
        path = dataset_root / f"{domain}.json"
        benchmark, prepare, score = load_domain(domain, path)
        cases = benchmark.all_observation_sets
        if not cases:
            raise ValueError(f"Empty dataset: {domain}")
        datasets[domain] = {"sha256": sha256(path.read_bytes()).hexdigest(), "cases": len(cases)}
        if quick:
            selected_id = quick_cases[domain]
            cases = [case for case in cases
                     if (case.get("observation_set_id") or case["observation_id"]) == selected_id]
            if len(cases) != 1:
                raise ValueError(f"Expected exactly one case for {domain}/{selected_id}")
        if medium:
            by_id = {(case.get("observation_set_id") or case["observation_id"]): case for case in cases}
            cases = [by_id[case_id] for case_id in medium_cases[domain]]
        for case in cases:
            observations, keys = prepare(case)
            if not keys:
                raise ValueError(f"Case without references: {domain}")
            case_id = case.get("observation_set_id") or case["observation_id"]
            prompt = benchmark.create_prompt(observations, [])
            jobs.append((domain, case_id, prompt, observations, keys, score,
                         args.queries or (len(keys) + 5 if early_stop else len(keys))))

    budget = sum(job[-1] for job in jobs)
    call_cap = budget * (2 if early_stop and tree else 1)
    print(f"{len(jobs)} cases, up to {budget} submissions and {call_cap} model calls", flush=True)
    if args.plan:
        return

    def make_agent(client):
        if args.method == "mcts-tree":
            return ReasoningTreeAgent(client, args.tree_policy, args.child_limit, args.duplicate_reward,
                                      rotate=args.exploration_list == "on")
        if args.method.startswith("mcts-"):
            return CoverageMCTSAgent(client, args.method.split("-", 1)[1])
        if args.method == "memory-structured":
            return StructuredMemoryAgent(client)
        return {"direct": DirectAgent, "memory": MemoryAgent, "mcts": MCTSAgent}[args.method](client)
    client = ArkCodingPlanClient(model=args.model)
    now = datetime.now(timezone.utc)
    mode = "medium" if medium else "quick" if quick else "baseline"
    if hard:
        mode = "hard_quick" if quick else "hard_full"
    label = f"{args.method}-list-{args.exploration_list}" if args.exploration_list else args.method
    output = ROOT / "results" / f"{label}_{mode}_{now:%Y%m%dT%H%M%S%fZ}.json"
    result = {"mode": mode, "agent": type(make_agent(client)).__name__, "history": args.method != "direct",
              "started_utc": now.isoformat(), "requested_model": client.model,
              "max_tokens": args.max_tokens, "queries_per_case": args.queries,
              "datasets": datasets, "expected_cases": len(jobs), "expected_requests": budget,
              "method": args.method, "call_cap": call_cap, "early_stop": early_stop,
              "completed": False, "cases": [], "summary": {}}

    if args.method == "mcts-tree":
        result["tree_config"] = {"policy": args.tree_policy, "child_limit": args.child_limit,
                                 "duplicate_reward": args.duplicate_reward}

    if args.exploration_list:
        result["tree_config"].update(exploration_list=args.exploration_list, branch_attribution=2)

    if args.resume:
        if args.method not in ("direct", "mcts", "mcts-tree"):
            parser.error("--resume supports direct, UCT and reasoning-tree runs")
        previous = json.loads(args.resume.read_text(encoding="utf-8"))
        fields = ("mode", "agent", "requested_model", "max_tokens", "queries_per_case",
                  "datasets", "expected_cases", "expected_requests")
        if args.method == "mcts-tree":
            fields += ("tree_config",)
        if previous["completed"] or any(previous[key] != result[key] for key in fields):
            parser.error("Resume file must be incomplete and match the current run settings")
        output = args.resume
        result = previous

    def save():
        output.parent.mkdir(exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    print(f"Saving results to {output.name}", flush=True)
    save()
    consecutive_errors = 0
    for index, (domain, case_id, prompt, observations, keys, score, queries) in enumerate(jobs, 1):
        agent = make_agent(client)
        agent.max_tokens = args.max_tokens
        records, unique, recovered = [], set(), set()
        submitted = 0
        search_cap = queries * (2 if early_stop and tree else 1)
        if index <= len(result["cases"]):
            continue
        partial = result.get("interrupted_case")
        if partial:
            normalize = lambda value: re.sub(r"(Use ONLY these operators: )([^\r\n]+)",
                                             lambda m: m[1] + ", ".join(sorted(m[2].split(", "))), value)
            if (partial["domain"], partial["case_id"], normalize(partial["prompt"])) != (domain, case_id, normalize(prompt)):
                raise ValueError("Interrupted case does not match the next case")
            prompt = partial["prompt"]
            records = list(partial["records"])
            for record in records:
                if "text" not in record:
                    continue
                answer = record["text"]
                if tree:
                    agent.next_prompt(prompt)
                    answer = agent.split_response(answer)[1]
                key, valid = score(answer, observations)
                if tree:
                    _, trace = agent.observe(record["text"], key, valid)
                    if trace.get("plan_path") != record.get("plan_path"):
                        raise ValueError("Cannot reconstruct the saved UCT path")
                if record.get("submitted", not early_stop) and key is not None:
                    unique.add(key)
                    if valid and key in keys:
                        recovered.add(key)
            submitted = sum(record.get("submitted", False) for record in records)
        started = datetime.now(timezone.utc)
        for call_index in range(len(records) + 1, search_cap + 1):
            if args.method == "mcts-tree":
                result["interrupted_case"] = {"domain": domain, "case_id": case_id,
                                              "prompt": prompt, "records": records}
                save()
            if isinstance(agent, ReasoningTreeAgent) and agent.exhausted:
                break
            if early_stop and (submitted >= queries or recovered == keys):
                break
            if hard and (call_index == 1 or call_index % 10 == 0):
                print(f"{args.method} {domain}/{case_id}: call {call_index}/{search_cap}, "
                      f"recovered {len(recovered)}/{len(keys)}", flush=True)
            wait_if_paused(args.pause_file)
            query_prompt = prompt if args.method == "direct" else agent.next_prompt(prompt)
            try:
                response = agent.solve(query_prompt)
            except LLMError as error:
                records.append({"error": str(error), "prompt": query_prompt, **error.details})
                consecutive_errors += 1
                if consecutive_errors >= 3 and (not early_stop or "no usable text" not in str(error)):
                    result["interrupted_case"] = {"domain": domain, "case_id": case_id,
                                                  "prompt": prompt, "records": records}
                    save()
                    raise SystemExit(f"Three consecutive request failures. Partial result: {output.name}")
                continue
            consecutive_errors = 0
            answer = response.text
            if isinstance(agent, CoverageMCTSAgent):
                answer = agent.answer_body(answer)
            if tree:
                answer = agent.split_response(answer)[1]
            key, valid = score(answer, observations) if answer.strip() else (None, False)
            strategy = {}
            if args.method != "direct":
                _, strategy = agent.observe(response.text, key, valid)
                strategy.update(prompt=query_prompt, answer=answer)
            is_submission = not early_stop or not tree or valid
            submitted += int(is_submission)
            records.append({**asdict(response), **strategy, "valid": valid,
                            "submitted": is_submission, "hypothesis_key": key,
                            "reference_hit": valid and key in keys})
            if is_submission and key is not None:
                unique.add(key)
                if valid and key in keys:
                    recovered.add(key)
        denominator = submitted if early_stop else queries
        result["cases"].append({
            "submission_cap": queries, "submitted": submitted, "model_call_cap": search_cap,
            "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "stop_reason": ("all_recovered" if recovered == keys and early_stop else
                            "submission_cap" if submitted >= queries else
                            "tree_exhausted" if isinstance(agent, ReasoningTreeAgent) and agent.exhausted
                            else "model_call_cap"),
            "domain": domain, "case_id": case_id, "prompt": prompt,
            "reference_count": len(keys), "records": records,
            "metrics": {"validity": sum(record.get("valid", False) for record in records) / max(denominator, 1),
                        "uniqueness": len(unique) / max(denominator, 1), "recovery": len(recovered) / len(keys)},
        })
        result.pop("interrupted_case", None)
        result["summary"] = summarize(result["cases"])
        save()
        print(f"{index}/{len(jobs)} {domain}/{case_id}", flush=True)

    result["completed"] = True
    save()
    print(json.dumps(result["summary"], indent=2))
