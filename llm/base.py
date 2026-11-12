"""Abstract client contract and shared response types."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """A provider request failed; messages never include credentials or bodies."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024) -> LLMResponse:
        """Return one response, or raise LLMError. Never treat errors as answers."""
        raise NotImplementedError
