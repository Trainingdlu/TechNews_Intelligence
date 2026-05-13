from __future__ import annotations

from pathlib import Path

from eval import build_task_dataset as mod
from services.llm_provider import DEFAULT_DEEPSEEK_MODEL


def _write_env(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _local_tmp_dir(tmp_path: Path) -> Path:
    path = tmp_path / "task_dataset_env_case"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_preparse_env_file_reads_path(tmp_path: Path) -> None:
    temp_dir = _local_tmp_dir(tmp_path)
    env_path = temp_dir / "task_eval.env"
    _write_env(env_path, ["TASK_EVAL_PROVIDER=deepseek"])
    args = mod._preparse_env_file(["--env-file", str(env_path)])
    assert args.env_file == env_path


def test_env_file_provider_and_model_apply_to_parse_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_dir = _local_tmp_dir(tmp_path)
    monkeypatch.delenv("TASK_EVAL_PROVIDER", raising=False)
    monkeypatch.delenv("TASK_EVAL_MODEL", raising=False)

    env_path = temp_dir / "task_eval.env"
    _write_env(
        env_path,
        [
            "TASK_EVAL_PROVIDER=deepseek",
            "TASK_EVAL_MODEL=deepseek-r1-distill-qwen-32b",
        ],
    )
    mod._load_eval_env(env_path)
    args = mod._parse_args([])
    assert args.provider == "deepseek"
    assert args.model == "deepseek-r1-distill-qwen-32b"


def test_cli_provider_and_model_override_env_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_dir = _local_tmp_dir(tmp_path)
    monkeypatch.delenv("TASK_EVAL_PROVIDER", raising=False)
    monkeypatch.delenv("TASK_EVAL_MODEL", raising=False)

    env_path = temp_dir / "task_eval.env"
    _write_env(
        env_path,
        [
            "TASK_EVAL_PROVIDER=deepseek",
            "TASK_EVAL_MODEL=deepseek-r1-distill-qwen-32b",
        ],
    )
    mod._load_eval_env(env_path)
    args = mod._parse_args(["--provider", "vertex", "--model", "gemini-3.1-pro-preview"])
    assert args.provider == "vertex"
    assert args.model == "gemini-3.1-pro-preview"


def test_provider_only_uses_provider_default_model(monkeypatch) -> None:
    monkeypatch.setenv("TASK_EVAL_PROVIDER", "deepseek")
    monkeypatch.delenv("TASK_EVAL_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    args = mod._parse_args([])

    assert args.provider == "deepseek"
    assert args.model == DEFAULT_DEEPSEEK_MODEL


def test_topic_audit_fail_override_defaults_to_blocking() -> None:
    args = mod._parse_args([])
    assert args.allow_topic_audit_fail is False

    args = mod._parse_args(["--allow-topic-audit-fail"])
    assert args.allow_topic_audit_fail is True
