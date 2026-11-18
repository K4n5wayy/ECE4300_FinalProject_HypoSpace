import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import direct_evaluation as evaluation
from llm import LLMResponse


class MediumEvaluationTests(unittest.TestCase):
    def test_invalid_search_does_not_submit_and_valid_duplicate_does(self):
        case = {"observation_id": "case"}
        benchmark = SimpleNamespace(
            all_observation_sets=[case],
            operators={"AND", "OR"},
            create_prompt=lambda *_: "Question",
        )
        answers = iter(["a", "invalid", "a", "b"] * 3)

        class Client:
            model = "mock"

            def generate(self, *args, **kwargs):
                return LLMResponse(next(answers), self.model, None, None, None, "stop")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            datasets.mkdir()
            domains = ("causal", "3d", "boolean")
            for domain in domains:
                (datasets / f"{domain}.json").write_text("{}")
            (datasets / "medium_cases.json").write_text(
                json.dumps({domain: ["case"] for domain in domains})
            )
            with (
                patch.object(evaluation, "ROOT", root),
                patch.object(evaluation, "ArkCodingPlanClient", return_value=Client()),
                patch.object(evaluation, "load_domain", return_value=(
                    benchmark,
                    lambda _: (None, {"a", "b"}),
                    lambda answer, _: (answer, answer != "invalid"),
                )),
                patch("sys.argv", ["test_medium", "--method", "mcts"]),
                patch("builtins.print"),
            ):
                evaluation.main(medium=True)
            result = json.loads(next((root / "results").glob("*.json")).read_text())
        self.assertTrue(result["completed"])
        for row in result["cases"]:
            self.assertEqual(row["submission_cap"], 7)
            self.assertEqual(row["submitted"], 3)
            self.assertEqual(len(row["records"]), 4)
            self.assertEqual(row["stop_reason"], "all_recovered")
            self.assertEqual(row["metrics"]["recovery"], 1)
