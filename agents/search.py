"""Per-case in-context memory and language-model tree search."""
from dataclasses import dataclass, field
from math import log, sqrt
import re

from .direct import DirectAgent


def memory_prompt(prompt, history):
    if not history:
        return prompt
    previous = "\n\n".join(history.values())
    return (
        f"{prompt}\n\nPreviously proposed hypotheses for this question:\n{previous}"
        "\n\nReturn a structurally different hypothesis that still satisfies every observation. "
        "Changing formatting or reordering parts of an equivalent answer does not count."
    )


class MemoryAgent(DirectAgent):
    def __init__(self, client):
        super().__init__(client)
        self.history = {}

    def next_prompt(self, prompt):
        return memory_prompt(prompt, self.history)

    def observe(self, text, key, valid):
        if key is not None:
            self.history.setdefault(key, text)
        return text, {}


@dataclass
class SearchNode:
    plan: str = ""
    visits: int = 0
    reward: float = 0.0
    children: dict = field(default_factory=dict)

    def uct(self, parent_visits):
        return self.reward / self.visits + sqrt(2 * log(parent_visits) / self.visits)


class MCTSAgent(MemoryAgent):
    """Search explicit plan prefixes using UCT and bundled expansion/rollout calls."""

    def __init__(self, client):
        super().__init__(client)
        self.root = SearchNode()
        self.valid_keys = set()
        self.path = []

    def select(self, node):
        return max(node.children.values(), key=lambda child: child.uct(node.visits))

    def next_prompt(self, prompt):
        node = self.root
        self.path = [node]
        # Progressive widening lets later calls revisit a useful plan or create a new one.
        while len(self.path) <= 2 and node.children:
            width = min(3, 1 + int(sqrt(node.visits / 2)))
            if len(node.children) < width:
                break
            node = self.select(node)
            self.path.append(node)

        prefix = " / ".join(node.plan for node in self.path[1:]) or "(start from observations)"
        siblings = " / ".join(child.plan for child in node.children.values()) or "(none)"
        task = memory_prompt(prompt, self.history)
        instruction = (
            f"Current search plan: {prefix}\n"
            f"Previously explored next steps at this node: {siblings}\n"
        )
        if len(self.path) <= 2:
            instruction += (
                "First write one line 'Plan: <a concrete new structural choice>' that extends "
                "the current plan and differs from the explored steps. Then carry that choice "
                "through to one complete answer on the following lines."
            )
        else:
            instruction += "Follow the current plan to produce a different complete answer."
        return f"{task}\n\n{instruction}\nUse the original answer format for the complete answer."

    @staticmethod
    def split_response(text):
        match = re.match(r"\s*Plan:\s*([^\r\n]+)\r?\n", text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), text[match.end():].strip()
        return "", text

    def observe(self, text, key, valid):
        plan, answer = self.split_response(text)
        parent = self.path[-1]
        if plan and len(self.path) <= 2:
            canonical_plan = " ".join(re.findall(r"\w+", plan.casefold()))
            if canonical_plan:
                self.path.append(parent.children.setdefault(canonical_plan, SearchNode(plan)))

        reward = float(valid and key is not None and key not in self.valid_keys)
        if valid and key is not None:
            self.valid_keys.add(key)
        for node in self.path:
            node.visits += 1
            node.reward += reward
        super().observe(answer, key, valid)
        return answer, {"plan_path": [node.plan for node in self.path[1:]], "reward": reward}


class StructuredMemoryAgent(MemoryAgent):
    """Focus generation on a concrete variation of a previously validated solution."""

    def next_prompt(self, prompt):
        if not self.history:
            return prompt
        direction = (
            "Start from a previously valid graph. Consider adding a shortcut between "
            "nodes already linked by a directed path, or removing an edge whose reachability "
            "is preserved by another path. Check the edge limit and observed reachability."
            if "directed acyclic graph" in prompt else
            "Keep the occupied columns fixed. Change the heights of one or more columns; "
            "check that the resulting height combination differs from all prior hypotheses."
            if "Top view:" in prompt else
            "Try a different outer operator or nesting pattern. Check the observations "
            "and depth limit. Changing operand order alone does not create a new mechanism."
        )
        return f"{memory_prompt(prompt, self.history)}\n\nSearch direction: {direction}"

    def observe(self, text, key, valid):
        if valid:
            super().observe(text, key, valid)
        return text, {}
