"""Shared chat-model provider factory for agent and eval runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_GEMINI_API_MODEL = "gemini-2.5-pro"
DEFAULT_VERTEX_MODEL = "gemini-3.1-pro-preview"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

GEMINI_API_PROVIDERS = {"gemini", "gemini_api", "google_ai_studio", "developer_api"}
VERTEX_PROVIDERS = {"vertex", "vertex_ai", "gcp"}
DEEPSEEK_PROVIDERS = {"deepseek", "deepseek_api", "deepseek_openai"}
SUPPORTED_PROVIDERS = GEMINI_API_PROVIDERS | VERTEX_PROVIDERS | DEEPSEEK_PROVIDERS


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = _clean(os.getenv(name, ""))
        if value:
            return value
    return default


def normalize_provider(provider: str | None, *, default_provider: str = "vertex") -> str:
    """Normalize provider aliases into one of: gemini_api, vertex, deepseek."""
    raw = _clean(provider) or _clean(default_provider)
    normalized = raw.lower()
    if normalized in GEMINI_API_PROVIDERS:
        return "gemini_api"
    if normalized in VERTEX_PROVIDERS:
        return "vertex"
    if normalized in DEEPSEEK_PROVIDERS:
        return "deepseek"
    allowed = "gemini_api|vertex|deepseek"
    raise ValueError(f"Unsupported provider: {provider}. Expected one of: {allowed}.")


def resolve_model_name(
    provider: str,
    model_name: str | None = None,
    *,
    default_model: str | None = None,
) -> str:
    """Resolve a model name from explicit input, provider env, then defaults."""
    normalized = normalize_provider(provider)
    explicit = _clean(model_name)
    if explicit:
        return explicit

    if normalized == "deepseek":
        return _first_env("DEEPSEEK_MODEL", default=DEFAULT_DEEPSEEK_MODEL)
    if normalized == "vertex":
        return _first_env(
            "VERTEX_GENERATION_MODEL",
            "VERTEX_MODEL",
            "GEMINI_MODEL",
            default=_clean(default_model) or DEFAULT_VERTEX_MODEL,
        )
    return _first_env("GEMINI_MODEL", default=_clean(default_model) or DEFAULT_GEMINI_API_MODEL)


def resolve_model_config(
    *,
    provider: str | None,
    model_name: str | None,
    default_provider: str = "vertex",
    default_model: str | None = None,
) -> ModelConfig:
    normalized = normalize_provider(provider, default_provider=default_provider)
    return ModelConfig(
        provider=normalized,
        model=resolve_model_name(normalized, model_name, default_model=default_model),
    )


def _deepseek_timeout() -> float:
    raw = _clean(os.getenv("DEEPSEEK_TIMEOUT_SEC", "120"))
    try:
        return max(5.0, min(float(raw), 600.0))
    except Exception:
        return 120.0


def build_chat_model(
    *,
    provider: str | None,
    model_name: str | None,
    temperature: float,
    default_provider: str = "vertex",
    default_model: str | None = None,
) -> Any:
    """Create a LangChain chat model for the requested provider."""
    config = resolve_model_config(
        provider=provider,
        model_name=model_name,
        default_provider=default_provider,
        default_model=default_model,
    )

    if config.provider == "gemini_api":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = _clean(os.getenv("GEMINI_API_KEY", ""))
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for provider=gemini_api.")
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=api_key,
            temperature=float(temperature),
        )

    if config.provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI

        project = _first_env("VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT is required for provider=vertex.")
        location = _first_env(
            "VERTEX_GENERATION_LOCATION",
            "VERTEX_LOCATION",
            "GOOGLE_CLOUD_LOCATION",
            default="global",
        )
        return ChatVertexAI(
            model=config.model,
            project=project,
            location=location,
            temperature=float(temperature),
        )

    api_key = _clean(os.getenv("DEEPSEEK_API_KEY", ""))
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for provider=deepseek.")
    base_url = _first_env("DEEPSEEK_BASE_URL", default=DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(temperature),
        timeout=_deepseek_timeout(),
    )


def resolve_agent_model_config() -> ModelConfig:
    """Resolve the provider/model used by the realtime agent runtime."""
    provider = normalize_provider(_clean(os.getenv("AGENT_MODEL_PROVIDER", "gemini_api")) or "gemini_api")
    explicit_model = _clean(os.getenv("AGENT_MODEL", ""))
    default_model = DEFAULT_GEMINI_API_MODEL
    if provider == "vertex":
        default_model = DEFAULT_VERTEX_MODEL
    elif provider == "deepseek":
        default_model = DEFAULT_DEEPSEEK_MODEL
    return resolve_model_config(
        provider=provider,
        model_name=explicit_model,
        default_provider="gemini_api",
        default_model=default_model,
    )


def build_agent_chat_model(*, temperature: float) -> Any:
    config = resolve_agent_model_config()
    return build_chat_model(
        provider=config.provider,
        model_name=config.model,
        temperature=temperature,
        default_provider="gemini_api",
        default_model=DEFAULT_GEMINI_API_MODEL,
    )


def agent_runtime_metadata() -> dict[str, str]:
    config = resolve_agent_model_config()
    return {
        "route": "custom_graph",
        "provider": config.provider,
        "model": config.model,
    }
