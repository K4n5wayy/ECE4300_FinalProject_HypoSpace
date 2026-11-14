"""Generate the fixed hard HypoSpace dataset with exhaustive reference sets."""
from collections import Counter
from itertools import combinations, permutations
import json
from math import comb
from pathlib import Path
from random import Random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "HypoSpace" / "_HypoSpace-main"
OUTPUT = ROOT / "datasets" / "hard"
SEED = 33550336
LIMIT = 30


def causal_dataset():
    nodes = list("ABCDEF")
    pairs = list(combinations(range(len(nodes)), 2))
    edges = []

    def enumerate_dags(index, descendants):
        if index == len(pairs) or len(edges) == 10:
            yield tuple(descendants), edges
            return

        yield from enumerate_dags(index + 1, descendants)
        left, right = pairs[index]
        for source, target in ((left, right), (right, left)):
            if descendants[target] & (1 << source):
                continue
            added = descendants[target] | (1 << target)
            updated = [mask | added if node == source or mask & (1 << source) else mask
                       for node, mask in enumerate(descendants)]
            edges.append((nodes[source], nodes[target]))
            yield from enumerate_dags(index + 1, updated)
            edges.pop()

    def graphs():
        return enumerate_dags(0, [0] * len(nodes))

    counts = Counter(signature for signature, _ in graphs())
    print(f"Counted {sum(counts.values())} DAGs over {len(counts)} observation sets", flush=True)
    signatures = []
    equivalent = set()
    for signature in sorted(counts, key=lambda signature: (-counts[signature], signature)):
        if signature in equivalent:
            continue
        signatures.append(signature)
        if len(signatures) == LIMIT:
            break
        for permutation in permutations(range(len(nodes))):
            equivalent.add(tuple(
                sum(1 << target for target, old_target in enumerate(permutation)
                    if signature[old_source] & (1 << old_target))
                for old_source in permutation
            ))
    selected = {signature: [] for signature in signatures}
    for signature, graph_edges in graphs():
        if signature in selected:
            selected[signature].append({"nodes": nodes, "edges": list(graph_edges)})

    cases = []
    for index, (signature, references) in enumerate(selected.items(), 1):
        observations = []
        for node, mask in zip(nodes, signature):
            effects = {target: int(bool(mask & (1 << i))) for i, target in enumerate(nodes)}
            formatted = " ".join(f"{target}:{effect}" for target, effect in effects.items())
            observations.append({"perturbed_node": node, "effects": effects,
                                 "string": f"Perturb({node}) -> {formatted}"})
        assert len(references) == counts[signature]
        cases.append({"observation_set_id": f"n6_hard_{index:03d}", "n_observations": 6,
                      "observations": observations, "ground_truth_graphs": references,
                      "n_compatible_graphs": len(references), "nodes": nodes, "max_edges": 10})
    return {"metadata": {"nodes": nodes, "max_edges": 10, "n_observations": 6,
                         "hypothesis_space_size": sum(counts.values()), "seed": SEED,
                         "n_datasets": len(cases), "complete_gt_coverage": True},
            "datasets": cases}


def generate_pool(script, filename, flags):
    path = OUTPUT / filename
    subprocess.run([sys.executable, str(UPSTREAM / script), *flags,
                    "--seed", str(SEED), "--output", str(path)], check=True)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    voxel = generate_pool("3d/generate_3d_dataset_complete.py", "3d_pool.json",
                          ["--grid-size", "3", "--max-height", "3", "--max-blocks", "3", "--fixed"])
    assert len(voxel["observation_sets"]) == comb(9, 3)
    voxel["observation_sets"] = sorted(
        Random(SEED).sample(voxel["observation_sets"], LIMIT), key=lambda row: row["observation_id"]
    )
    voxel["metadata"].update(n_observation_sets=LIMIT, seed=SEED)
    boolean = generate_pool("boolean/boolean_dataset.py", "boolean_pool.json",
                            ["--operators", "full", "--max-depth", "2"])
    cases = [row for group in boolean["datasets_by_n_observations"].values() for row in group]
    cases.sort(key=lambda row: (-row["n_compatible_expressions"], row["observation_set_id"]))
    groups = {}
    for case in cases[:LIMIT]:
        groups.setdefault(str(case["n_observations"]), []).append(case)
    boolean["datasets_by_n_observations"] = groups
    boolean["metadata"].update(total_observation_sets=sum(map(len, groups.values())), seed=SEED)
    for domain, data in (("3d", voxel), ("boolean", boolean), ("causal", None)):
        if data is None:
            data = causal_dataset()
        path = OUTPUT / f"{domain}.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
