#!/usr/bin/env bash
set -euo pipefail

# Value-profile checkpoint-resume eval pipeline.
# Goal:
# - Reuse existing progress from eval/datasets/task_eval_v1_cases.jsonl.checkpoint.json
# - Run a high-value subset of task types
# - Complete dataset -> matrix (2 groups) -> leaderboard

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/TechNews_Intelligence}"
DEPLOY_DIR="${PROJECT_DIR}/deployment"
ENV_FILE="${DEPLOY_DIR}/.env"
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.yml"

PROFILE="${PROFILE:-balanced}"
case "${PROFILE}" in
  balanced)
    PROFILE_TASK_FILE_REL="eval/config/task_types_value_budget_balanced.json"
    PROFILE_POOLS_PER_TASK_DEFAULT="10"
    ;;
  compact)
    PROFILE_TASK_FILE_REL="eval/config/task_types_value_budget_compact.json"
    PROFILE_POOLS_PER_TASK_DEFAULT="8"
    ;;
  *)
    echo "[ValueEval][Error] PROFILE must be one of: balanced, compact. got=${PROFILE}" >&2
    exit 1
    ;;
esac

TASK_FILE_REL="${TASK_FILE_REL:-${PROFILE_TASK_FILE_REL}}"
TASK_FILE="${PROJECT_DIR}/${TASK_FILE_REL}"

case "${TASK_FILE_REL}" in
  "eval/config/task_types_value_budget_balanced.json"|"eval/config/task_types_value_budget_compact.json")
    ;;
  *)
    echo "[ValueEval][Error] TASK_FILE_REL only supports optimized profiles: task_types_value_budget_balanced.json or task_types_value_budget_compact.json" >&2
    exit 1
    ;;
esac

SRC_CHECKPOINT_REL="eval/datasets/task_eval_v1_cases.jsonl.checkpoint.json"
SRC_CHECKPOINT="${PROJECT_DIR}/${SRC_CHECKPOINT_REL}"
CHECKPOINT_REL="${CHECKPOINT_REL:-eval/datasets/task_eval_v1_${PROFILE}_cases.checkpoint.json}"
CHECKPOINT="${PROJECT_DIR}/${CHECKPOINT_REL}"

DATASET_REL="${DATASET_REL:-eval/datasets/task_eval_v1_${PROFILE}_cases.jsonl}"
DATASET="${PROJECT_DIR}/${DATASET_REL}"
MANIFEST_REL="${MANIFEST_REL:-eval/datasets/task_eval_v1_${PROFILE}_manifest.json}"
MANIFEST="${PROJECT_DIR}/${MANIFEST_REL}"

RUNS_PER_CASE="${RUNS_PER_CASE:-1}"
MATRIX_GROUPS="${MATRIX_GROUPS:-G0_baseline,G5_full_optimized}"
MATRIX_SLEEP_SECONDS="${MATRIX_SLEEP_SECONDS:-5}"

BUILD_POOLS_PER_TASK="${BUILD_POOLS_PER_TASK:-${PROFILE_POOLS_PER_TASK_DEFAULT}}"
BUILD_LLM_MAX_RETRIES="${BUILD_LLM_MAX_RETRIES:-4}"
BUILD_LLM_BACKOFF_SEC="${BUILD_LLM_BACKOFF_SEC:-8}"
BUILD_AUDIT_MAX_REGEN_ROUNDS="${BUILD_AUDIT_MAX_REGEN_ROUNDS:-8}"
BUILD_POOLS_PER_GENERATION_CALL="${BUILD_POOLS_PER_GENERATION_CALL:-1}"
BUILD_CASES_PER_AUDIT_CALL="${BUILD_CASES_PER_AUDIT_CALL:-4}"
BUILD_INTER_LLM_CALL_SLEEP_SEC="${BUILD_INTER_LLM_CALL_SLEEP_SEC:-60}"
BUILD_INTER_TASK_SLEEP_SEC="${BUILD_INTER_TASK_SLEEP_SEC:-15}"

PROVIDER="${PROVIDER:-vertex}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"

VERSION_FILE="${VERSION_FILE:-/tmp/technews_value_${PROFILE}_version.txt}"
RUN_ID="${RUN_ID:-value_${PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}"
REPORT_DIR_REL="eval/reports/${RUN_ID}"
REPORT_DIR="${PROJECT_DIR}/${REPORT_DIR_REL}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ValueEval][Error] missing env file: ${ENV_FILE}" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "[ValueEval][Error] neither docker compose nor docker-compose is available." >&2
  exit 1
fi

compose() {
  (
    cd "${DEPLOY_DIR}"
    "${COMPOSE_CMD[@]}" --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
  )
}

echo "[ValueEval] project=${PROJECT_DIR}"
echo "[ValueEval] run_id=${RUN_ID}"
echo "[ValueEval] profile=${PROFILE}"
echo "[ValueEval] provider=${PROVIDER} model=${MODEL}"
echo "[ValueEval] runs_per_case=${RUNS_PER_CASE} matrix_groups=${MATRIX_GROUPS}"

mkdir -p "${PROJECT_DIR}/eval/config" "${PROJECT_DIR}/eval/datasets" "${PROJECT_DIR}/eval/reports"

# 0) Strong guards: ensure code contains required fixes.
grep -q "failed rc=" "${PROJECT_DIR}/deployment/scripts/eval/run_skill_matrix_pipeline.sh" || {
  echo "[ValueEval][Error] missing step() failure fix (failed rc=) in run_skill_matrix_pipeline.sh" >&2
  exit 1
}
grep -q "audit-max-regen-rounds" "${PROJECT_DIR}/eval/build_task_dataset_v1.py" || {
  echo "[ValueEval][Error] missing audit-max-regen-rounds in build_task_dataset_v1.py" >&2
  exit 1
}
grep -q "runner_exit_0_but_report_missing_or_invalid" "${PROJECT_DIR}/eval/run_matrix_eval.py" || {
  echo "[ValueEval][Error] missing matrix report integrity guard in run_matrix_eval.py" >&2
  exit 1
}

# 1) Stop residual processes/containers.
pkill -f "run_skill_matrix_pipeline.sh|eval.build_task_dataset_v1|eval.run_matrix_eval|eval.build_task_eval_v1_leaderboard|eval.run_task_eval_v1" || true
docker ps --format '{{.ID}} {{.Command}}' \
  | grep -E "eval.build_task_dataset_v1|run_skill_matrix_pipeline|eval.run_matrix_eval|eval.run_task_eval_v1|eval.build_task_eval_v1_leaderboard" \
  | awk '{print $1}' | xargs -r docker kill || true

# 2) Validate task file.
if [[ ! -f "${TASK_FILE}" ]]; then
  echo "[ValueEval][Error] profile task file not found: ${TASK_FILE}" >&2
  exit 1
fi
echo "[ValueEval] using profile task file: ${TASK_FILE_REL}"

# 3) Build dedicated checkpoint by reusing existing progress.
if [[ ! -f "${SRC_CHECKPOINT}" ]]; then
  echo "[ValueEval][Error] source checkpoint not found: ${SRC_CHECKPOINT}" >&2
  exit 1
fi

PROJECT_DIR_PY="${PROJECT_DIR}" SRC_CHECKPOINT_REL_PY="${SRC_CHECKPOINT_REL}" CHECKPOINT_REL_PY="${CHECKPOINT_REL}" TASK_FILE_REL_PY="${TASK_FILE_REL}" DATASET_REL_PY="${DATASET_REL}" python3 - <<'PY'
import json
import os
import pathlib

project_dir = pathlib.Path(os.environ["PROJECT_DIR_PY"]).resolve()
src = project_dir / os.environ["SRC_CHECKPOINT_REL_PY"]
dst = project_dir / os.environ["CHECKPOINT_REL_PY"]

d = json.loads(src.read_text(encoding="utf-8"))
d["task_type_file"] = str((project_dir / os.environ["TASK_FILE_REL_PY"]).resolve())
d["dataset_path"] = str((project_dir / os.environ["DATASET_REL_PY"]).resolve())

dst.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("[ValueEval] checkpoint reused:")
print("  status=", d.get("status"))
print("  completed_tasks=", len(d.get("completed_task_ids", [])))
print("  cases_buffered=", len(d.get("cases", [])))
print("  checkpoint=", dst)
PY

# 4) Ensure DB up.
compose up -d postgres

# 5) Build dataset (resume from dedicated checkpoint).
(
  cd "${PROJECT_DIR}"
  compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${PROJECT_DIR}/eval/datasets:/app/eval/datasets" \
    -v "${PROJECT_DIR}/eval/config:/app/eval/config" \
    bot python -u -m eval.build_task_dataset_v1 \
      --task-types "${TASK_FILE_REL}" \
      --output "${DATASET_REL}" \
      --manifest-output "${MANIFEST_REL}" \
      --checkpoint-path "${CHECKPOINT_REL}" \
      --resume-from-checkpoint \
      --no-enforce-coverage-policy \
      --provider "${PROVIDER}" \
      --model "${MODEL}" \
      --temperature "${TEMPERATURE}" \
      --seed "${SEED}" \
      --pools-per-task "${BUILD_POOLS_PER_TASK}" \
      --llm-max-retries "${BUILD_LLM_MAX_RETRIES}" \
      --llm-backoff-sec "${BUILD_LLM_BACKOFF_SEC}" \
      --audit-max-regen-rounds "${BUILD_AUDIT_MAX_REGEN_ROUNDS}" \
      --pools-per-generation-call "${BUILD_POOLS_PER_GENERATION_CALL}" \
      --cases-per-audit-call "${BUILD_CASES_PER_AUDIT_CALL}" \
      --inter-llm-call-sleep-sec "${BUILD_INTER_LLM_CALL_SLEEP_SEC}" \
      --inter-task-sleep-sec "${BUILD_INTER_TASK_SLEEP_SEC}"
)

if [[ ! -s "${DATASET}" ]]; then
  echo "[ValueEval][Error] dataset not generated: ${DATASET}" >&2
  exit 1
fi
if [[ ! -s "${MANIFEST}" ]]; then
  echo "[ValueEval][Error] manifest not generated: ${MANIFEST}" >&2
  exit 1
fi

# 6) Freeze dataset version.
V="v_value_${PROFILE}_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_DIR}/eval/datasets/versions/${V}"
cp "${DATASET}" "${PROJECT_DIR}/eval/datasets/versions/${V}/regression.jsonl"
cp "${DATASET}" "${PROJECT_DIR}/eval/datasets/versions/${V}/smoke.jsonl"
cp "${MANIFEST}" "${PROJECT_DIR}/eval/datasets/versions/${V}/manifest.json"
echo "${V}" | tee "${VERSION_FILE}" >/dev/null
echo "[ValueEval] frozen version=${V}"

# 7) Run matrix (2 groups, 1 run per case).
mkdir -p "${REPORT_DIR}"
(
  cd "${PROJECT_DIR}"
  compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${PROJECT_DIR}/eval/datasets:/app/eval/datasets" \
    -v "${PROJECT_DIR}/eval/reports:/app/eval/reports" \
    bot python -u -m eval.run_matrix_eval \
      --matrix eval/experiment_matrix.json \
      --groups "${MATRIX_GROUPS}" \
      --output-dir "${REPORT_DIR_REL}/matrix" \
      -- \
      --dataset "eval/datasets/versions/${V}/regression.jsonl" \
      --runs-per-case "${RUNS_PER_CASE}" \
      --sleep-seconds "${MATRIX_SLEEP_SECONDS}" \
      --include-trace-summary
)

MANIFEST_PATH="$(ls -t "${REPORT_DIR}/matrix/"*_manifest.json | head -n1 || true)"
if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "[ValueEval][Error] matrix manifest missing under ${REPORT_DIR}/matrix" >&2
  exit 1
fi

MATRIX_MANIFEST_PATH="${MANIFEST_PATH}" MATRIX_GROUPS_ENV="${MATRIX_GROUPS}" python3 - <<'PY'
import json
import os
import pathlib

manifest_path = pathlib.Path(os.environ["MATRIX_MANIFEST_PATH"])
if not manifest_path.exists() or manifest_path.stat().st_size <= 0:
  raise SystemExit(f"[ValueEval][Error] matrix manifest invalid: {manifest_path}")

data = json.loads(manifest_path.read_text(encoding="utf-8"))
failed_groups = int(data.get("failed_groups", 0) or 0)
if failed_groups > 0:
  raise SystemExit(f"[ValueEval][Error] matrix failed_groups={failed_groups}; manifest={manifest_path}")

requested = [x.strip() for x in os.environ.get("MATRIX_GROUPS_ENV", "").split(",") if x.strip()]
groups = data.get("groups", [])
if not isinstance(groups, list) or not groups:
  raise SystemExit(f"[ValueEval][Error] matrix groups missing in manifest: {manifest_path}")

group_map = {
  str(g.get("id", "")).strip(): pathlib.Path(str(g.get("output", "")).strip())
  for g in groups
  if isinstance(g, dict)
}

missing_groups = [gid for gid in requested if gid not in group_map]
if missing_groups:
  raise SystemExit(
    f"[ValueEval][Error] requested groups missing in manifest: {missing_groups}; manifest={manifest_path}"
  )

missing_outputs = []
for gid in requested:
  out = group_map.get(gid)
  if out is None or (not out.exists()) or out.stat().st_size <= 0:
    missing_outputs.append({"group": gid, "output": str(out) if out else ""})

if missing_outputs:
  raise SystemExit(
    f"[ValueEval][Error] group outputs missing/empty: {missing_outputs}; manifest={manifest_path}"
  )

print(
  f"[ValueEval] matrix integrity ok groups={len(requested)} failed_groups={failed_groups} manifest={manifest_path}"
)
PY

# 8) Build leaderboard.
(
  cd "${PROJECT_DIR}"
  compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${PROJECT_DIR}/eval/reports:/app/eval/reports" \
    bot python -u -m eval.build_task_eval_v1_leaderboard \
      --manifest "eval/reports/${RUN_ID}/matrix/$(basename "${MANIFEST_PATH}")" \
      --baseline-group G0_baseline \
      --output-json "eval/reports/${RUN_ID}/leaderboard/latest.json" \
      --output-md "eval/reports/${RUN_ID}/leaderboard/latest.md"
)

LEADERBOARD_JSON="${PROJECT_DIR}/eval/reports/${RUN_ID}/leaderboard/latest.json"
LEADERBOARD_MD="${PROJECT_DIR}/eval/reports/${RUN_ID}/leaderboard/latest.md"
if [[ ! -s "${LEADERBOARD_JSON}" ]]; then
  echo "[ValueEval][Error] leaderboard JSON missing/empty: ${LEADERBOARD_JSON}" >&2
  exit 1
fi
if [[ ! -s "${LEADERBOARD_MD}" ]]; then
  echo "[ValueEval][Error] leaderboard markdown missing/empty: ${LEADERBOARD_MD}" >&2
  exit 1
fi

LEADERBOARD_JSON_PATH="${LEADERBOARD_JSON}" python3 - <<'PY'
import json
import os
import pathlib

path = pathlib.Path(os.environ["LEADERBOARD_JSON_PATH"])
data = json.loads(path.read_text(encoding="utf-8"))
groups = data.get("groups", [])
if not isinstance(groups, list) or not groups:
    raise SystemExit(f"[ValueEval][Error] leaderboard groups missing: {path}")
print(f"[ValueEval] leaderboard integrity ok groups={len(groups)} path={path}")
PY

echo "[ValueEval] done"
echo "[ValueEval] run_id=${RUN_ID}"
echo "[ValueEval] dataset_version=${V}"
echo "[ValueEval] matrix_manifest=eval/reports/${RUN_ID}/matrix/$(basename "${MANIFEST_PATH}")"
echo "[ValueEval] leaderboard=eval/reports/${RUN_ID}/leaderboard/latest.json"
