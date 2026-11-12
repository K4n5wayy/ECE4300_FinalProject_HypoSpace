"""One controlled case using upstream enumeration, prompt and validation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HypoSpace" / "_HypoSpace-main" / "causal"))
from run_causal_benchmark import CausalBenchmarkEnhanced
from generate_causal_dataset import CausalDatasetGenerator, PerturbationObservation


def build_case():
    benchmark = CausalBenchmarkEnhanced()
    benchmark.nodes = ["A", "B", "C"]
    benchmark.max_edges = 3
    observation = PerturbationObservation("A", {"A": 0, "B": 1, "C": 1})
    observations = [{"perturbed_node": "A", "effects": observation.effects,
                     "string": observation.to_string()}]
    graphs = CausalDatasetGenerator.generate_all_dags(benchmark.nodes, max_edges=3)
    valid = [g for g in graphs if benchmark.validate_hypothesis(g, observations)]
    # References are held by the evaluator only; never sent to the agent.
    return benchmark, observations, valid
