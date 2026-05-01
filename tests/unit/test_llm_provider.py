from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from services.llm_provider import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    agent_runtime_metadata,
    build_chat_model,
    normalize_provider,
    resolve_model_config,
)


class _FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, attr: str) -> None:
    module = types.ModuleType(name)
    setattr(module, attr, _FakeChatModel)
    monkeypatch.setitem(sys.modules, name, module)


def test_normalize_provider_accepts_deepseek_aliases() -> None:
    assert normalize_provider("deepseek") == "deepseek"
    assert normalize_provider("deepseek_api") == "deepseek"
    assert normalize_provider("deepseek_openai") == "deepseek"


def test_deepseek_builder_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_chat_model(provider="deepseek", model_name="deepseek-v4-pro", temperature=0.0)


def test_deepseek_builder_uses_base_url_and_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    model = build_chat_model(provider="deepseek", model_name="", temperature=0.2)

    assert isinstance(model, _FakeChatModel)
    assert model.kwargs["model"] == DEFAULT_DEEPSEEK_MODEL
    assert model.kwargs["api_key"] == "sk-test"
    assert model.kwargs["base_url"] == DEFAULT_DEEPSEEK_BASE_URL
    assert model.kwargs["temperature"] == 0.2


def test_gemini_and_vertex_builders_use_shared_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "langchain_google_genai", "ChatGoogleGenerativeAI")
    _install_fake_module(monkeypatch, "langchain_google_vertexai", "ChatVertexAI")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("VERTEX_PROJECT", "vertex-project")

    gemini = build_chat_model(provider="gemini_api", model_name="gemini-test", temperature=0.1)
    vertex = build_chat_model(provider="vertex", model_name="gemini-vertex-test", temperature=0.0)

    assert gemini.kwargs["model"] == "gemini-test"
    assert gemini.kwargs["google_api_key"] == "gemini-key"
    assert vertex.kwargs["model"] == "gemini-vertex-test"
    assert vertex.kwargs["project"] == "vertex-project"


def test_agent_runtime_metadata_uses_agent_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODEL_PROVIDER", "deepseek_api")
    monkeypatch.setenv("AGENT_MODEL", "deepseek-v4-pro")

    assert agent_runtime_metadata() == {
        "route": "custom_graph",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
    }


def test_eval_provider_config_can_resolve_deepseek_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    config = resolve_model_config(
        provider="deepseek",
        model_name=None,
        default_provider="vertex",
        default_model="gemini-3.1-pro-preview",
    )

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-pro"


def test_run_eval_script_uses_role_level_model_variables() -> None:
    script = Path("deployment/scripts/eval/run_eval.sh").read_text(encoding="utf-8")

    assert 'PROVIDER="${PROVIDER:-' not in script
    assert 'MODEL="${MODEL:-' not in script
    assert 'AGENT_PROVIDER="${AGENT_PROVIDER:-vertex}"' in script
    assert 'DATASET_PROVIDER="${DATASET_PROVIDER:-vertex}"' in script
    assert 'JUDGE_PROVIDER="${JUDGE_PROVIDER:-vertex}"' in script
