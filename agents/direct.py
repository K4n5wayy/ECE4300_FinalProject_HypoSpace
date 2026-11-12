"""Baseline: one model call, no tools, memory, planning or refinement."""
from llm import LLMClient, LLMResponse
from .base import BaseAgent


class DirectAgent(BaseAgent):
    def __init__(self, client: LLMClient, *, max_tokens: int = 2048):
        self.client = client
        self.max_tokens = max_tokens

    def solve(self, prompt: str) -> LLMResponse:
        return self.client.generate(prompt, max_tokens=self.max_tokens)
