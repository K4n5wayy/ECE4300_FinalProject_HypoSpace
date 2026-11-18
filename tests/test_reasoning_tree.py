import unittest

from agents.reasoning_tree import Node, ReasoningTreeAgent


class ReasoningTreeTests(unittest.TestCase):
    def test_duplicate_step_is_not_added_and_duplicate_answer_is_penalized(self):
        agent = ReasoningTreeAgent(None, "uct", 5, -1)
        agent.next_prompt("Question")
        _, first = agent.observe("Plan: Try an edge.\nGraph: A -> B", "edge", True)
        self.assertEqual(first["reward"], 1)
        agent.next_prompt("Question")
        self.assertIn("Try an edge.", agent.describe([agent.root]))
        agent.path = [agent.root]
        _, duplicate = agent.observe("Plan: try an edge\nGraph: A -> B", "edge", True)
        self.assertEqual(duplicate["tree_event"], "duplicate_step")
        self.assertEqual(duplicate["reward"], -1)
        self.assertEqual(len(agent.root.children), 1)

    def test_full_node_solves_and_exhausted_node_closes(self):
        agent = ReasoningTreeAgent(None, "uct", 5, 0)
        agent.root.children = {str(i): Node(str(i)) for i in range(5)}
        self.assertIn("Mode: SOLVE", agent.next_prompt("Question"))
        agent.observe("Plan: Another step\nCandidate", None, False)
        self.assertEqual(len(agent.root.children), 5)
        agent.root.children.clear()
        agent.next_prompt("Question")
        agent.observe("EXHAUSTED", None, False)
        self.assertTrue(agent.exhausted)

class ExplorationListTests(unittest.TestCase):
    def test_failed_expansion_defers_node_until_available_nodes_are_tried(self):
        agent = ReasoningTreeAgent(None, "llm", 7, -1, rotate=True)
        agent.next_prompt("Question")
        agent.observe("Branch: 0\nPlan: try NOR\nCandidate", None, False)
        agent.next_prompt("Question")
        agent.observe("Branch: 0\nPlan: try NOR\nCandidate", None, False)
        agent.next_prompt("Question")
        self.assertEqual(len(agent.options), 1)
        self.assertEqual(agent.options[0][-1].plan, "try NOR")
        agent.observe("Branch: 0\nCandidate", None, False)
        agent.next_prompt("Question")
        self.assertEqual(agent.round, 2)
        self.assertEqual(len(agent.options), 2)

    def test_single_visible_solve_node_is_credited_without_branch_label(self):
        agent = ReasoningTreeAgent(None, "llm", 7, -1, rotate=True)
        child = Node("Try shortcuts", children={str(i): Node(str(i)) for i in range(7)})
        agent.root.children["shortcuts"] = child
        agent.deferred.add(id(agent.root))
        agent.next_prompt("Question")
        _, details = agent.observe("Graph: A -> B", "new", True)
        self.assertEqual(details["tree_event"], "solve")
        self.assertEqual(child.reward, 1)
        self.assertEqual(agent.deferred, {id(agent.root)})

    def test_ambiguous_answer_updates_history_without_crediting_a_node(self):
        agent = ReasoningTreeAgent(None, "llm", 7, -1, rotate=True)
        agent.root.children["other"] = Node("Other")
        agent.next_prompt("Question")
        _, details = agent.observe("Graph: A -> B", "new", True)
        self.assertEqual(details["tree_event"], "unconfirmed_branch")
        self.assertEqual(agent.root.visits, 0)
        self.assertFalse(agent.deferred)
        self.assertIn("new", agent.history)

    def test_reflection_reaches_next_prompt_without_entering_the_answer(self):
        agent = ReasoningTreeAgent(None, "llm", 7, -1, branch_examples=True, reflect=True)
        self.assertIn("Reflection:", agent.next_prompt("Question"))
        response = "Branch: 0\nPlan: Try a conjunction\nExpression: x AND y\nReflection: Try another outer operator."
        self.assertEqual(agent.split_response(response)[1], "Expression: x AND y")
        agent.observe(response, "new", True)
        self.assertIn("Last self-review: Try another outer operator.", agent.next_prompt("Question"))
        self.assertEqual(agent.history["new"], "Expression: x AND y")
