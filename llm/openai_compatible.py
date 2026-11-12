"""HTTP adapter for OpenAI-compatible chat completions."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .base import LLMClient, LLMError, LLMResponse


class OpenAICompatibleClient(LLMClient):
    """Adapter for providers implementing the chat/completions wire format."""

    def __init__(self, *, api_key: str, model: str, base_url: str,
                 timeout: float = 120):
        if not api_key or not api_key.strip():
            raise ValueError("API key is required")
        if not model or not model.strip():
            raise ValueError("Model is required")
        parsed = urlparse(base_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Base URL must be an HTTPS URL without credentials/query/fragment")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self._api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024) -> LLMResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be nonempty text")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "stream": False}
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key,
                     "Content-Type": "application/json"}, method="POST")
        try:
            # No automatic retries: repeated requests can consume extra quota.
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            status = error.code
            error.close()
            raise LLMError(f"Model service returned HTTP {status}; check credentials, model and quota") from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("Model connection failed or timed out") from None
        except (ValueError, UnicodeError):
            raise LLMError("Model service returned invalid JSON") from None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError()
            usage = data.get("usage") or {}
            return LLMResponse(
                text=content, model=data.get("model") or self.model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                finish_reason=choice.get("finish_reason"))
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise LLMError("Model response has no usable text or an unexpected structure") from None
