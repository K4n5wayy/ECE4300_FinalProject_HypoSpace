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

    def test_nested_nor_matches_reference_structure(self):
        benchmark, prepare, score = load_domain("boolean", ROOT / "datasets/hard/boolean.json")
        case = next(row for row in benchmark.all_observation_sets
                    if row["observation_set_id"] == "n1_inputs_00_outputs_0")
        observations, keys = prepare(case)
        for operand in ("x", "y", "NOT x", "NOT y", "x AND y", "x OR y"):
            with self.subTest(operand=operand):
                actual = score(f"Expression: NOR({operand}, NOR(x,y))", observations)
                expected = score(f"Expression: NOT (({operand}) OR NOT (x OR y))", observations)
                self.assertEqual(actual, expected)
                self.assertTrue(actual[1])
                self.assertIn(actual[0], keys)
        self.assertNotEqual(score("Expression: NOT(NOT x)", observations)[0],
                            score("Expression: x", observations)[0])
        self.assertEqual(score("Expression: NOR(x, NOR(x,y)", observations), (None, False))

    def test_assistance_requires_verbatim_evidence_and_runs_only_on_parse_failure(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from experiments.boolean_scoring import assist_boolean_score

        score = Mock(side_effect=[(None, False), ("double-not", True)])
        client = Mock()
        client.generate.return_value = SimpleNamespace(text='{"expression":"NOT(NOT x)"}')
        answer = "My final answer is NOT(NOT x)."
        self.assertEqual(assist_boolean_score(answer, [], score, client), ("double-not", True))
        score.assert_called_with("Expression: NOT(NOT x)", [])
        score = Mock(return_value=(None, False))
        client.generate.return_value.text = '{"expression":"y"}'
        self.assertEqual(assist_boolean_score(answer, [], score, client), (None, False))
        score.assert_called_once()
        client.reset_mock()
        score = Mock(return_value=("parsed-but-invalid", False))
        self.assertEqual(assist_boolean_score(answer, [], score, client), ("parsed-but-invalid", False))
        client.generate.assert_not_called()
