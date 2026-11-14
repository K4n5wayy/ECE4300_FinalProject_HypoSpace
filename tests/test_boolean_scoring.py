import unittest

from experiments.direct_evaluation import ROOT, load_domain


class BooleanScoringTests(unittest.TestCase):
    def test_reference_formulas_roundtrip_in_both_datasets(self):
        for dataset in ("boolean.json", "hard/boolean.json"):
            benchmark, prepare, score = load_domain("boolean", ROOT / "datasets" / dataset)
            for case in benchmark.all_observation_sets:
                observations, keys = prepare(case)
                recovered = set()
                for reference in case["ground_truth_expressions"]:
                    key, valid = score("Expression: " + reference["formula"], observations)
                    self.assertTrue(valid, reference["formula"])
                    recovered.add(key)
                self.assertEqual(recovered, keys)

    def test_nor_and_unenumerated_expression(self):
        benchmark, prepare, score = load_domain("boolean", ROOT / "datasets/hard/boolean.json")
        case = next(row for row in benchmark.all_observation_sets
                    if row["observation_set_id"] == "n1_inputs_00_outputs_0")
        observations, keys = prepare(case)
        key, valid = score("Expression: NOR(NOT x, y)", observations)
        self.assertTrue(valid)
        self.assertIn(key, keys)
        self.assertFalse(score("Expression: x AND NOT y AND NOT x", observations)[1])
        self.assertFalse(score("Expression: NOR(x, y)", observations)[1])
