#!/usr/bin/env bash
set -euo pipefail

# One-shot v1 pipeline:
# 1) Build task-driven skill dataset (v1)
# 2) Freeze dataset version under eval/datasets/versions/
# 3) Run matrix eval (runner = task_eval_v1) on frozen regression.jsonl
# 4) Build task_eval_v1 leaderboard report
#
# Host requirements:
# - docker + docker compose
# - deployment/.env configured
#
# Usage examples:
#   bash deployment/scripts/eval/run_skill_matrix_pipeline.sh
#   RUN_ID=full_20260419T120000Z DATASET_VERSION=v_task_20260419_120000 \
#     bash deployment/scripts/eval/run_skill_matrix_pipeline.sh
#   GROUPS=G0_baseline,G5_full_optimized RUNS_PER_CASE=1 \
#     bash deployment/scripts/eval/run_skill_matrix_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deployment"
ENV_FILE="${DEPLOY_DIR}/.env"
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.yml"
LOCK_DIR="${REPO_ROOT}/.locks"
LOCK_FILE="${LOCK_DIR}/run_skill_matrix_pipeline.lock"

RUN_ID="${RUN_ID:-skill_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_VERSION="${DATASET_VERSION:-v_task_$(date -u +%Y%m%d_%H%M%S)}"
PROVIDER="${PROVIDER:-vertex}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
RUNS_PER_CASE="${RUNS_PER_CASE:-3}"
MATRIX_SLEEP_SECONDS="${MATRIX_SLEEP_SECONDS:-20}"
GROUPS="${GROUPS:-}"
BASELINE_GROUP="${BASELINE_GROUP:-G0_baseline}"
MATRIX_FILE="${MATRIX_FILE:-eval/experiment_matrix.json}"
BUILD_LLM_MAX_RETRIES="${BUILD_LLM_MAX_RETRIES:-2}"
BUILD_LLM_BACKOFF_SEC="${BUILD_LLM_BACKOFF_SEC:-2}"
BUILD_INTER_TASK_SLEEP_SEC="${BUILD_INTER_TASK_SLEEP_SEC:-0}"
BUILD_RESUME_FROM_CHECKPOINT="${BUILD_RESUME_FROM_CHECKPOINT:-1}"
BUILD_POOLS_PER_GENERATION_CALL="${BUILD_POOLS_PER_GENERATION_CALL:-1}"
BUILD_CASES_PER_AUDIT_CALL="${BUILD_CASES_PER_AUDIT_CALL:-4}"
BUILD_INTER_LLM_CALL_SLEEP_SEC="${BUILD_INTER_LLM_CALL_SLEEP_SEC:-60}"
BUILD_STEP_MAX_ATTEMPTS="${BUILD_STEP_MAX_ATTEMPTS:-20}"
BUILD_STEP_RETRY_SLEEP_SEC="${BUILD_STEP_RETRY_SLEEP_SEC:-300}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy from deployment/.env.example first." >&2
  exit 1
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "[SkillMatrix][Error] 'flock' is required for single-instance locking." >&2
  exit 1
fi

mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[SkillMatrix][Error] another pipeline is already running. lock=${LOCK_FILE}" >&2
  exit 2
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Neither 'docker compose' nor 'docker-compose' is available." >&2
  exit 1
fi

compose() {
  (
    cd "${DEPLOY_DIR}"
    "${COMPOSE_CMD[@]}" --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
  )
}

step() {
  local name="$1"
  shift
  echo "[SkillMatrix][Step] ${name} started"
  "$@"
  echo "[SkillMatrix][Step] ${name} done"
}

echo "[SkillMatrix] repo=${REPO_ROOT}"
echo "[SkillMatrix] run_id=${RUN_ID}"
echo "[SkillMatrix] dataset_version=${DATASET_VERSION}"
echo "[SkillMatrix] provider=${PROVIDER} model=${MODEL} runs_per_case=${RUNS_PER_CASE}"
echo "[SkillMatrix] matrix_sleep_seconds=${MATRIX_SLEEP_SECONDS}"
echo "[SkillMatrix] matrix_file=${MATRIX_FILE}"
echo "[SkillMatrix] build_llm_max_retries=${BUILD_LLM_MAX_RETRIES} build_llm_backoff_sec=${BUILD_LLM_BACKOFF_SEC}"
echo "[SkillMatrix] build_inter_task_sleep_sec=${BUILD_INTER_TASK_SLEEP_SEC}"
echo "[SkillMatrix] build_resume_from_checkpoint=${BUILD_RESUME_FROM_CHECKPOINT}"
echo "[SkillMatrix] build_pools_per_generation_call=${BUILD_POOLS_PER_GENERATION_CALL} build_cases_per_audit_call=${BUILD_CASES_PER_AUDIT_CALL}"
echo "[SkillMatrix] build_inter_llm_call_sleep_sec=${BUILD_INTER_LLM_CALL_SLEEP_SEC}"
echo "[SkillMatrix] build_step_max_attempts=${BUILD_STEP_MAX_ATTEMPTS} build_step_retry_sleep_sec=${BUILD_STEP_RETRY_SLEEP_SEC}"

mkdir -p "${REPO_ROOT}/eval/reports/${RUN_ID}"
mkdir -p "${REPO_ROOT}/eval/datasets/versions"

# Ensure DB is up for retrieval-backed eval.
step "postgres_up" compose up -d postgres

# Step 1: build task dataset.
BUILD_DATASET_ARGS=(
  --task-types eval/config/task_types_v1.json
  --output eval/datasets/task_eval_v1_cases.jsonl
  --manifest-output eval/datasets/task_eval_v1_manifest.json
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --llm-max-retries "${BUILD_LLM_MAX_RETRIES}"
  --llm-backoff-sec "${BUILD_LLM_BACKOFF_SEC}"
  --inter-task-sleep-sec "${BUILD_INTER_TASK_SLEEP_SEC}"
  --pools-per-generation-call "${BUILD_POOLS_PER_GENERATION_CALL}"
  --cases-per-audit-call "${BUILD_CASES_PER_AUDIT_CALL}"
  --inter-llm-call-sleep-sec "${BUILD_INTER_LLM_CALL_SLEEP_SEC}"
)
if [[ "${BUILD_RESUME_FROM_CHECKPOINT}" == "0" ]]; then
  BUILD_DATASET_ARGS+=(--no-resume-from-checkpoint)
fi

BUILD_ATTEMPT=1
BUILD_OK=0
while [[ "${BUILD_ATTEMPT}" -le "${BUILD_STEP_MAX_ATTEMPTS}" ]]; do
  echo "[SkillMatrix][BuildRetry] attempt=${BUILD_ATTEMPT}/${BUILD_STEP_MAX_ATTEMPTS}"
  if step "build_task_dataset_v1" compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${REPO_ROOT}/eval/datasets:/app/eval/datasets" \
    bot python -u -m eval.build_task_dataset_v1 \
      "${BUILD_DATASET_ARGS[@]}"; then
    BUILD_OK=1
    break
  fi
  if [[ "${BUILD_ATTEMPT}" -ge "${BUILD_STEP_MAX_ATTEMPTS}" ]]; then
    break
  fi
  echo "[SkillMatrix][BuildRetry] attempt=${BUILD_ATTEMPT} failed; sleep ${BUILD_STEP_RETRY_SLEEP_SEC}s then retry."
  sleep "${BUILD_STEP_RETRY_SLEEP_SEC}"
  BUILD_ATTEMPT=$((BUILD_ATTEMPT + 1))
done

if [[ "${BUILD_OK}" -ne 1 ]]; then
  echo "[SkillMatrix][Error] build_task_dataset_v1 failed after ${BUILD_STEP_MAX_ATTEMPTS} attempts." >&2
  exit 6
fi
if [[ ! -s "${REPO_ROOT}/eval/datasets/task_eval_v1_cases.jsonl" ]]; then
  echo "[SkillMatrix][Error] missing eval/datasets/task_eval_v1_cases.jsonl after Step 1." >&2
  exit 3
fi
if [[ ! -s "${REPO_ROOT}/eval/datasets/task_eval_v1_manifest.json" ]]; then
  echo "[SkillMatrix][Error] missing eval/datasets/task_eval_v1_manifest.json after Step 1." >&2
  exit 3
fi

# Step 2: freeze dataset version.
step "freeze_dataset_version" compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval/datasets:/app/eval/datasets" \
  bot sh -lc "
    set -euo pipefail
    V='${DATASET_VERSION}'
    mkdir -p /app/eval/datasets/versions/\"\${V}\"
    cp /app/eval/datasets/task_eval_v1_cases.jsonl /app/eval/datasets/versions/\"\${V}\"/regression.jsonl
    cp /app/eval/datasets/task_eval_v1_cases.jsonl /app/eval/datasets/versions/\"\${V}\"/smoke.jsonl
    cp /app/eval/datasets/task_eval_v1_manifest.json /app/eval/datasets/versions/\"\${V}\"/manifest.json
    echo \"[SkillMatrix] frozen dataset -> /app/eval/datasets/versions/\${V}\"
  "
if [[ ! -s "${REPO_ROOT}/eval/datasets/versions/${DATASET_VERSION}/regression.jsonl" ]]; then
  echo "[SkillMatrix][Error] missing frozen regression dataset after Step 2." >&2
  exit 4
fi

# Step 3: matrix eval.
MATRIX_ARGS=(
  --matrix "${MATRIX_FILE}"
  --output-dir "eval/reports/${RUN_ID}/matrix"
)
if [[ -n "${GROUPS}" ]]; then
  MATRIX_ARGS+=(--groups "${GROUPS}")
fi

step "run_matrix_eval" compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -e AGENT_MODEL_PROVIDER="${PROVIDER}" \
  -e GEMINI_MODEL="${MODEL}" \
  -e VERTEX_GENERATION_MODEL="${MODEL}" \
  -e VERTEX_MODEL="${MODEL}" \
  -v "${REPO_ROOT}/eval/datasets:/app/eval/datasets" \
  -v "${REPO_ROOT}/eval/reports:/app/eval/reports" \
  bot python -u -m eval.run_matrix_eval \
    "${MATRIX_ARGS[@]}" \
    -- \
    --dataset "eval/datasets/versions/${DATASET_VERSION}/regression.jsonl" \
    --runs-per-case "${RUNS_PER_CASE}" \
    --sleep-seconds "${MATRIX_SLEEP_SECONDS}" \
    --include-trace-summary

MANIFEST_PATH="$(ls -t "${REPO_ROOT}/eval/reports/${RUN_ID}/matrix/"*_manifest.json | head -n1 || true)"
if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "[SkillMatrix][Error] matrix manifest not found under eval/reports/${RUN_ID}/matrix" >&2
  exit 1
fi
MANIFEST_REL="eval/reports/${RUN_ID}/matrix/$(basename "${MANIFEST_PATH}")"

# Step 4: v1 leaderboard.
step "build_task_eval_v1_leaderboard" compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval/reports:/app/eval/reports" \
  bot python -u -m eval.build_task_eval_v1_leaderboard \
    --manifest "${MANIFEST_REL}" \
    --baseline-group "${BASELINE_GROUP}" \
    --output-json "eval/reports/${RUN_ID}/leaderboard/latest.json" \
    --output-md "eval/reports/${RUN_ID}/leaderboard/latest.md"
if [[ ! -s "${REPO_ROOT}/eval/reports/${RUN_ID}/leaderboard/latest.json" ]]; then
  echo "[SkillMatrix][Error] missing leaderboard JSON after Step 4." >&2
  exit 5
fi

echo "[SkillMatrix] done"
echo "[SkillMatrix] run_id=${RUN_ID}"
echo "[SkillMatrix] dataset_version=${DATASET_VERSION}"
echo "[SkillMatrix] matrix_manifest=${MANIFEST_REL}"
echo "[SkillMatrix] leaderboard=eval/reports/${RUN_ID}/leaderboard/latest.json"
