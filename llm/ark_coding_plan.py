"""Volcengine Coding Plan configuration and client."""
import json
import os
from pathlib import Path
from .openai_compatible import OpenAICompatibleClient


ANSWER_MODEL = "deepseek-v4.1-flash-economy"


class ArkCodingPlanClient(OpenAICompatibleClient):
    """Volcengine Coding Plan; deliberately separate from the metered Ark URL."""
    BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"

    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 timeout: float | None = None, config_path: str | Path | None = None):
        path = (Path(config_path) if config_path is not None else
                Path(__file__).resolve().parents[1] / "config" / "llm.json")
        config = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("ark_coding_plan"), dict):
                raise ValueError()
            config = data["ark_coding_plan"]
            for field in ("api_key", "model"):
                if field in config and not isinstance(config[field], str):
                    raise ValueError()
            if "timeout" in config and (isinstance(config["timeout"], bool) or
                    not isinstance(config["timeout"], (int, float)) or
                    not 0 < config["timeout"] < float("inf")):
                raise ValueError()
        except FileNotFoundError:
            if config_path is not None:
                raise ValueError("指定的模型配置文件不存在") from None
        except (ValueError, UnicodeError):
            raise ValueError("模型配置无效：请检查 JSON 格式和 ark_coding_plan 字段类型") from None
        except OSError:
            raise ValueError("无法读取模型配置文件") from None
        # Explicit arguments > local configuration > environment > defaults.
        key = api_key if api_key is not None else config.get("api_key") or os.getenv("ARK_API_KEY", "")
        if not key.strip():
            raise ValueError("请在 config/llm.json 中填写 api_key，或设置 ARK_API_KEY")
        super().__init__(
            api_key=key,
            model=model or config.get("model") or os.getenv("ARK_MODEL") or ANSWER_MODEL,
            base_url=self.BASE_URL,
            timeout=timeout if timeout is not None else config.get("timeout", 120))
