"""Ragas scoring + optional LangSmith dataset upload for eval reports."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.lstrip("\ufeff").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _extract_contexts_from_trace_summary(summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    contexts: list[str] = []
    seen: set[str] = set()
    for event in summary.get("tool_events", []) or []:
        if not isinstance(event, dict):
            continue
        output_summary = event.get("output_summary", {})
        if not isinstance(output_summary, dict):
            continue
        docs = output_summary.get("context_docs", [])
        if not isinstance(docs, list):
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            snippet = str(doc.get("summary") or doc.get("title") or doc.get("url") or "").strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            contexts.append(snippet)
    return contexts


def build_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiment = report.get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}
    experiment_group = str(experiment.get("group", "")).strip()

    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        outputs = case.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            continue
        answer = str(outputs[0]).strip()
        if not answer or answer.startswith("[EVAL_ERROR]"):
            continue

        constraints = case.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        reference = str(constraints.get("ground_truth", "")).strip()
        if not reference:
            expected_facts = constraints.get("expected_facts", [])
            if isinstance(expected_facts, list):
                tokens = [str(item).strip() for item in expected_facts if str(item).strip()]
                reference = "；".join(tokens)

        runs = case.get("runs", [])
        trace_summary: dict[str, Any] | None = None
        if isinstance(runs, list) and runs:
            first_run = runs[0]
            if isinstance(first_run, dict):
                candidate = first_run.get("trace_summary")
                if isinstance(candidate, dict):
                    trace_summary = candidate
        contexts = _extract_contexts_from_trace_summary(trace_summary)
        if not contexts:
            ragas_contexts = constraints.get("ragas_contexts", [])
            if isinstance(ragas_contexts, list):
                contexts = [str(item).strip() for item in ragas_contexts if str(item).strip()]

        rows.append(
            {
                "case_id": str(case.get("id", "")).strip(),
                "question": str(case.get("question", "")).strip(),
                "answer": answer,
                "reference": reference,
                "contexts": contexts,
                "experiment_group": experiment_group,
            }
        )
    return rows


def _build_ragas_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        contexts = row.get("contexts", [])
        if not isinstance(contexts, list):
            contexts = []
        contexts = [str(item).strip() for item in contexts if str(item).strip()]
        reference = str(row.get("reference", "")).strip()
        if not question or not answer:
            continue
        payloads.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "reference": reference,
                "ground_truth": reference,
            }
        )
    return payloads


def _build_ragas_runtime():
    provider = os.getenv("AGENT_MODEL_PROVIDER", "gemini_api").strip().lower()
    if provider in {"vertex", "vertex_ai", "gcp"}:
        from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

        llm = ChatVertexAI(
            model_name=os.getenv("VERTEX_MODEL", "gemini-3.1-pro-preview").strip(),
            temperature=0.0,
        )
        embeddings = VertexAIEmbeddings(
            model_name=os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005").strip()
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip(),
            temperature=0.0,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004").strip()
        )

    return llm, embeddings


def run_ragas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    ragas_rows = _build_ragas_dataset_rows(rows)
    if not ragas_rows:
        return {
            "rows": [],
            "summary": {},
            "metric_names": [],
        }

    llm, embeddings = _build_ragas_runtime()
    dataset = Dataset.from_list(ragas_rows)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )

    if hasattr(result, "to_pandas"):
        table = result.to_pandas().to_dict(orient="records")
    elif hasattr(result, "to_dict"):
        table = result.to_dict(orient="records")  # type: ignore[call-arg]
    else:
        raise RuntimeError("Unsupported ragas result object: missing to_pandas/to_dict")

    metric_names = [name for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")]
    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        payload = dict(row)
        metric_record = table[idx] if idx < len(table) and isinstance(table[idx], dict) else {}
        scores: dict[str, float] = {}
        for metric in metric_names:
            value = metric_record.get(metric)
            try:
                if value is not None:
                    scores[metric] = float(value)
            except Exception:
                continue
        payload["scores"] = scores
        scored_rows.append(payload)

    summary: dict[str, float] = {}
    for metric in metric_names:
        values = [row.get("scores", {}).get(metric) for row in scored_rows]
        filtered = [float(v) for v in values if isinstance(v, (int, float))]
        if filtered:
            summary[metric] = sum(filtered) / len(filtered)

    return {
        "rows": scored_rows,
        "summary": summary,
        "metric_names": metric_names,
    }


def _get_or_create_dataset(client: Any, dataset_name: str, description: str) -> Any:
    try:
        iterator = client.list_datasets(dataset_name=dataset_name)
    except TypeError:
        iterator = client.list_datasets()
    for dataset in iterator:
        if str(getattr(dataset, "name", "")) == dataset_name:
            return dataset
    return client.create_dataset(dataset_name=dataset_name, description=description)


def upload_rows_to_langsmith(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    from langsmith import Client

    client = Client()
    dataset = _get_or_create_dataset(
        client,
        dataset_name=dataset_name,
        description="TechNews eval rows with optional Ragas scores.",
    )
    dataset_id = str(getattr(dataset, "id"))
    created = 0
    for row in rows:
        inputs = {
            "question": str(row.get("question", "")).strip(),
            "contexts": row.get("contexts", []),
        }
        outputs = {
            "answer": str(row.get("answer", "")).strip(),
            "reference": str(row.get("reference", "")).strip(),
        }
        metadata = {
            "case_id": str(row.get("case_id", "")).strip(),
            "experiment_group": str(row.get("experiment_group", "")).strip(),
            "scores": row.get("scores", {}),
        }
        client.create_example(
            dataset_id=dataset_id,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        created += 1
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "examples_created": created,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ragas scoring and optional LangSmith upload.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Eval report JSON from eval/run_eval.py (requires outputs in report).",
    )
    parser.add_argument(
        "--rows-jsonl",
        type=Path,
        default=None,
        help="Prebuilt Ragas rows JSONL (alternative to --report).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reports/ragas/latest.json"),
        help="Output JSON path for scored rows/summary.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip ragas scoring and only export/upload rows.",
    )
    parser.add_argument(
        "--upload-langsmith",
        action="store_true",
        help="Upload rows (with scores if available) to LangSmith dataset.",
    )
    parser.add_argument(
        "--langsmith-dataset",
        type=str,
        default="",
        help="LangSmith dataset name (default: technews-ragas-<timestamp>).",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.report and not args.rows_jsonl:
        raise ValueError("One of --report or --rows-jsonl is required.")

    rows: list[dict[str, Any]]
    source_info: dict[str, Any] = {}
    if args.rows_jsonl:
        jsonl_path = args.rows_jsonl.resolve()
        rows = _load_jsonl(jsonl_path)
        source_info["rows_jsonl"] = str(jsonl_path)
    else:
        report_path = args.report.resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = build_rows_from_report(report)
        source_info["report"] = str(report_path)

    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_info,
        "row_count": len(rows),
        "rows": rows,
        "summary": {},
    }

    if not args.skip_ragas:
        ragas_result = run_ragas(rows)
        result["rows"] = ragas_result["rows"]
        result["summary"] = ragas_result["summary"]
        result["metric_names"] = ragas_result["metric_names"]

    if args.upload_langsmith:
        dataset_name = args.langsmith_dataset.strip()
        if not dataset_name:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dataset_name = f"technews-ragas-{ts}"
        upload_result = upload_rows_to_langsmith(result["rows"], dataset_name)
        result["langsmith"] = upload_result

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Ragas] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
