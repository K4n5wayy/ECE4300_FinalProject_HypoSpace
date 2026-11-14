import unittest

from agents.coverage_search import CoverageMCTSAgent
from agents.search import MCTSAgent, MemoryAgent, SearchNode


class SearchTests(unittest.TestCase):
    def test_breadth_and_recent_reward_can_choose_different_branches(self):
        older = SearchNode("older", visits=3, reward=2)
        newer = SearchNode("newer", visits=4, reward=2)
        root = SearchNode(visits=7, children={"older": older, "newer": newer})
        self.assertIs(CoverageMCTSAgent(None, "breadth").select(root), older)
        recent = CoverageMCTSAgent(None, "recent")
        recent.recent = {id(older): 0, id(newer): 0.9}
        self.assertIs(recent.select(root), newer)

    def test_llm_branch_selection_uses_existing_node_and_preserves_answer(self):
        agent = CoverageMCTSAgent(None, "llm")
        agent.next_prompt("Question")
        agent.observe("Plan: direct edge\nGraph: A -> B", "a", True)
        self.assertIn("1. direct edge", agent.next_prompt("Question"))
        answer, details = agent.observe(
            "Branch: 1\nPlan: add another edge\nGraph: A -> B, B -> C", "b", True
        )
        self.assertEqual(answer, "Graph: A -> B, B -> C")
        self.assertEqual(details["selected_branch"], 1)
        self.assertEqual(details["plan_path"], ["direct edge", "add another edge"])

    def test_memory_keeps_one_example_per_hypothesis_and_resets_per_case(self):
        agent = MemoryAgent(None)
        self.assertEqual(agent.next_prompt("Question"), "Question")
        agent.observe("Graph: A -> B", "same", True)
        agent.observe("Graph: A → B", "same", True)
        prompt = agent.next_prompt("Question")
        self.assertIn("Graph: A -> B", prompt)
        self.assertNotIn("Graph: A → B", prompt)
        self.assertEqual(MemoryAgent(None).next_prompt("Question"), "Question")

    def test_mcts_updates_ancestors_and_rewards_only_new_valid_hypotheses(self):
        agent = MCTSAgent(None)
        agent.next_prompt("Question")
        text = "Plan: use a direct edge\nGraph: A -> B"
        answer, first = agent.observe(text, "edge", True)
        self.assertEqual(answer, "Graph: A -> B")
        self.assertEqual(first["reward"], 1)
        self.assertIn("Current search plan: use a direct edge", agent.next_prompt("Question"))
        _, duplicate = agent.observe("Plan: keep B a sink\nGraph: A -> B", "edge", True)
        self.assertEqual(duplicate["reward"], 0)
        self.assertEqual(agent.root.visits, 2)
        self.assertEqual(agent.root.reward, 1)
        self.assertEqual(agent.root.children["use a direct edge"].visits, 2)
        self.assertIn("Previously explored next steps at this node: use a direct edge",
                      agent.next_prompt("Question"))
        _, invalid = agent.observe("Plan: try another route\nGraph: X -> B", None, False)
        self.assertEqual(invalid["reward"], 0)


if __name__ == "__main__":
    unittest.main()
