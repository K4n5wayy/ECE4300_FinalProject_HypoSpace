"""Search natural-language reasoning steps with explicit expansion and closure."""
from dataclasses import dataclass, field
from math import log, sqrt
import re

from .search import MemoryAgent, memory_prompt


@dataclass
class Node:
    plan: str = ""
    children: dict = field(default_factory=dict)
    visits: int = 0
    reward: float = 0.0
    recent: float = 0.0
    closed: bool = False


class ReasoningTreeAgent(MemoryAgent):
    def __init__(self, client, policy, child_limit, duplicate_reward, *, rotate=False):
        super().__init__(client)
        self.root = Node()
        self.policy = policy
        self.child_limit = child_limit
        self.duplicate_reward = duplicate_reward
        self.valid_keys = set()
        self.options = []
        self.rotate = rotate
        self.deferred = set()
        self.round = 1

    @property
    def exhausted(self):
        return self.root.closed

    def select(self, parent, candidates):
        if self.policy == "breadth":
            return min(candidates, key=lambda node: node.visits)
        def priority(node):
            if not node.visits:
                return float("inf")
            value = node.recent if self.policy == "recent" else node.reward / node.visits
            return value + sqrt(2 * log(max(parent.visits, 1)) / node.visits)
        return max(candidates, key=priority)

    def next_prompt(self, prompt):
        self.path = [self.root]
        while len(self.path) <= 2:
            node = self.path[-1]
            live = [child for child in node.children.values() if not child.closed]
            if not live or node.visits % 2 == 0:
                break
            self.path.append(self.select(node, live))

        self.options = []
        node = self.path[-1]
        task = memory_prompt(prompt, self.history)
        if self.policy == "llm":
            self.path = [self.root]
            self.options = [[self.root]] + [
                [self.root, child] for child in self.root.children.values() if not child.closed
            ]
            if self.rotate:
                available = [path for path in self.options if id(path[-1]) not in self.deferred]
                if not available:
                    self.deferred.clear()
                    self.round += 1
                else:
                    self.options = available
            choices = "\n".join(
                f"{i}: {self.describe(path)} (recent reward {path[-1].recent:.2f})"
                for i, path in enumerate(self.options)
            )
            return (f"{task}\n\nChoose the direction most likely to produce a new valid answer. "
                    f"Start with Branch: <number>.\n{choices}\n\n{self.protocol()}")
        return f"{task}\n\n{self.describe(self.path)}\n\n{self.protocol()}"

    def describe(self, path):
        node = path[-1]
        plans = " / ".join(part.plan for part in path[1:]) or "Start from the observations"
        children = " | ".join(child.plan for child in node.children.values()) or "(none)"
        mode = "SOLVE" if self.must_solve(path) else "EXPAND"
        return f"{plans}. Mode: {mode}. Existing child steps: {children}."

    def must_solve(self, path):
        return len(path[-1].children) >= self.child_limit or len(path) > 2

    @staticmethod
    def protocol():
        return (
            "In EXPAND mode, write Plan: followed by one new logical step distinct in meaning "
            "from every existing child step, then one complete candidate following that step. "
            "If this direction cannot be expanded, write EXHAUSTED on its own line. "
            "If you already have a candidate, include it after EXHAUSTED. "
            "In SOLVE mode, give one complete candidate directly following the selected path. "
            "Use the original answer format for the complete candidate."
        )

    @staticmethod
    def split_response(text):
        body = re.sub(r"^\s*Branch:[ \t]*\d+[ \t]*\r?\n", "", text, count=1, flags=re.I)
        first, _, answer = body.strip().partition("\n")
        if first.strip().casefold() == "exhausted":
            return "exhausted", answer.strip()
        if first.casefold().startswith("plan:"):
            return first[5:].strip(), answer.strip()
        return "", body

    def observe(self, text, key, valid):
        plan, answer = self.split_response(text)
        confirmed = True
        if self.options:
            match = re.match(r"\s*Branch:[ \t]*(\d{1,3})[ \t]*\r?\n", text, re.I)
            if match and int(match.group(1)) < len(self.options):
                self.path = self.options[int(match.group(1))]
            elif len(self.options) == 1:
                self.path = self.options[0]
            else:
                self.path = []
                confirmed = False
        event = "solve" if confirmed else "unconfirmed_branch"
        if confirmed and not self.must_solve(self.path):
            node = self.path[-1]
            if plan == "exhausted":
                node.closed = True
                event = "closed"
            elif plan:
                canonical = " ".join(re.findall(r"\w+", plan.casefold()))
                if canonical and canonical not in node.children:
                    child = Node(plan)
                    node.children[canonical] = child
                    self.path = [*self.path, child]
                    event = "expanded"
                else:
                    event = "duplicate_step"
            elif self.rotate:
                event = "missing_step"
            if self.rotate and event in ("duplicate_step", "missing_step"):
                self.deferred.add(id(node))

        reward = 0
        if valid and key is not None:
            reward = self.duplicate_reward if key in self.valid_keys else 1
            self.valid_keys.add(key)
            super().observe(answer, key, valid)
        for member in self.path:
            member.visits += 1
            member.reward += reward
            member.recent = 0.5 * (reward + member.recent)
        return answer, {"plan_path": [member.plan for member in self.path[1:]],
                        "reward": reward, "tree_event": event, "exploration_round": self.round}
