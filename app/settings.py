from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env(path: Path | None = None) -> None:
    """Load simple key-value settings from the ignored project `.env` file."""

    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class ChatProviderConfig:
    """Resolved OpenAI-compatible chat provider settings."""

    provider: str
    api_key: str
    model: str
    base_url: str | None = None


def get_chat_provider() -> ChatProviderConfig | None:
    """Resolve OpenAI or Qwen credentials without exposing their values."""

    load_project_env()
    if os.getenv("SCIENTIFIC_AGENT_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return None
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "qwen":
        api_key = os.getenv("QWEN_API_KEY", "").strip()
        if not api_key:
            return None
        return ChatProviderConfig(
            provider="qwen",
            api_key=api_key,
            model=os.getenv("QWEN_MODEL", "qwen3.8-27b"),
            base_url=os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            ),
        )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return ChatProviderConfig(
        provider="openai",
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


def create_chat_client():
    """Create the configured OpenAI-compatible SDK client."""

    from openai import OpenAI

    config = get_chat_provider()
    if config is None:
        raise RuntimeError("No chat provider API key is configured.")
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
    )


def provider_feature_enabled(feature: str) -> bool:
    """Check whether a configured model should handle one optional workflow role."""

    config = get_chat_provider()
    if config is None:
        return False
    configured = os.getenv("LLM_FEATURES", "planning,critic,synthesis,grounding").lower().split(",")
    return feature.lower() in {item.strip() for item in configured}
