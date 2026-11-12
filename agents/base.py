"""Common interface for benchmark-solving strategies."""
from abc import ABC, abstractmethod
from llm import LLMResponse


class BaseAgent(ABC):
    @abstractmethod
    def solve(self, prompt: str) -> LLMResponse:
        """Produce one candidate without receiving the reference solutions."""
        raise NotImplementedError
