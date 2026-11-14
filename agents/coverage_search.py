"""Coverage-aware variants of the existing plan tree."""
from math import log, sqrt
import re

from .search import MCTSAgent


class CoverageMCTSAgent(MCTSAgent):
    def __init__(self, client, policy):
        super().__init__(client)
        self.policy = policy
        self.recent = {}

    def select(self, node):
        children = list(node.children.values())
        if self.policy == "breadth":
            return min(children, key=lambda child: child.visits)
        if self.policy == "recent":
            def priority(child):
                mean_reward = self.recent.get(id(child), 0)
                return mean_reward + sqrt(2 * log(node.visits) / child.visits)

            return max(children, key=priority)
        return max(children, key=lambda child: child.uct(node.visits))

    def next_prompt(self, prompt):
        self.options = []
        task = super().next_prompt(prompt)
        if self.policy != "llm" or not self.root.children:
            return task
        self.options = list(self.root.children.values())
        choices = "\n".join(
            f"{i}. {node.plan} (visits={node.visits}, recent novelty="
            f"{self.recent.get(id(node), 0):.2f})"
            for i, node in enumerate(self.options, 1)
        )
        from .search import memory_prompt

        return (
            f"{memory_prompt(prompt, self.history)}\n\nAvailable search branches:\n{choices}\n"
            "Choose a branch likely to yield a different valid answer, or 0 for a new direction. "
            "Output 'Branch: <number>' on the first line, 'Plan: <concrete structural choice>' "
            "on the second, then one complete answer in the original format. "
            "Follow the selected branch when proposing the answer."
        )

    @staticmethod
    def answer_body(text):
        return re.sub(r"^\s*Branch:[ \t]*\d+[ \t]*\r?\n", "", text, count=1, flags=re.I)

    def observe(self, text, key, valid):
        branch = re.match(r"\s*Branch:[ \t]*(\d{1,3})[ \t]*\r?\n", text, re.I)
        selected = int(branch.group(1)) if branch else 0
        if self.options:
            self.path = [self.root]
            if 1 <= selected <= len(self.options):
                self.path.append(self.options[selected - 1])
        answer, details = super().observe(self.answer_body(text), key, valid)
        for node in self.path:
            node_id = id(node)
            self.recent[node_id] = 0.5 * details["reward"] + 0.5 * self.recent.get(node_id, 0)
        if self.options:
            details["selected_branch"] = selected if branch and selected <= len(self.options) else None
        return answer, details
