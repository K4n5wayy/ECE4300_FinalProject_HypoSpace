"""Canonical configurations for the hard three-question tree sweep."""
from itertools import product

POLICIES = ("uct", "breadth", "recent", "llm")
CHILD_LIMITS = (5, 7)
DUPLICATE_REWARDS = (0, -1)

# Keep the breadth runs with more saved progress. Breadth ignores rewards.
CONFIGURATIONS = tuple(
    (policy, limit, reward)
    for policy, limit, reward in product(POLICIES, CHILD_LIMITS, DUPLICATE_REWARDS)
    if policy != "breadth" or reward == -1
)
