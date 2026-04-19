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

RUN_ID="${RUN_ID:-skill_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_VERSION="${DATASET_VERSION:-v_task_$(date -u +%Y%m%d_%H%M%S)}"
PROVIDER="${PROVIDER:-vertex}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
RUNS_PER_CASE="${RUNS_PER_CASE:-3}"
GROUPS="${GROUPS:-}"
BASELINE_GROUP="${BASELINE_GROUP:-G0_baseline}"
MATRIX_FILE="${MATRIX_FILE:-eval/experiment_matrix.json}"
BUILD_LLM_MAX_RETRIES="${BUILD_LLM_MAX_RETRIES:-2}"
BUILD_LLM_BACKOFF_SEC="${BUILD_LLM_BACKOFF_SEC:-2}"
BUILD_INTER_TASK_SLEEP_SEC="${BUILD_INTER_TASK_SLEEP_SEC:-0}"
BUILD_RESUME_FROM_CHECKPOINT="${BUILD_RESUME_FROM_CHECKPOINT:-1}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy from deployment/.env.example first." >&2
  exit 1
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

echo "[SkillMatrix] repo=${REPO_ROOT}"
echo "[SkillMatrix] run_id=${RUN_ID}"
echo "[SkillMatrix] dataset_version=${DATASET_VERSION}"
echo "[SkillMatrix] provider=${PROVIDER} model=${MODEL} runs_per_case=${RUNS_PER_CASE}"
echo "[SkillMatrix] matrix_file=${MATRIX_FILE}"
echo "[SkillMatrix] build_llm_max_retries=${BUILD_LLM_MAX_RETRIES} build_llm_backoff_sec=${BUILD_LLM_BACKOFF_SEC}"
echo "[SkillMatrix] build_inter_task_sleep_sec=${BUILD_INTER_TASK_SLEEP_SEC}"
echo "[SkillMatrix] build_resume_from_checkpoint=${BUILD_RESUME_FROM_CHECKPOINT}"

mkdir -p "${REPO_ROOT}/eval/reports/${RUN_ID}"
mkdir -p "${REPO_ROOT}/eval/datasets/versions"

# Ensure DB is up for retrieval-backed eval.
compose up -d postgres

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
)
if [[ "${BUILD_RESUME_FROM_CHECKPOINT}" == "0" ]]; then
  BUILD_DATASET_ARGS+=(--no-resume-from-checkpoint)
fi

compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval/datasets:/app/eval/datasets" \
  bot python -u -m eval.build_task_dataset_v1 \
    "${BUILD_DATASET_ARGS[@]}"

# Step 2: freeze dataset version.
compose run --rm --no-deps \
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

# Step 3: matrix eval.
MATRIX_ARGS=(
  --matrix "${MATRIX_FILE}"
  --output-dir "eval/reports/${RUN_ID}/matrix"
)
if [[ -n "${GROUPS}" ]]; then
  MATRIX_ARGS+=(--groups "${GROUPS}")
fi

compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval/datasets:/app/eval/datasets" \
  -v "${REPO_ROOT}/eval/reports:/app/eval/reports" \
  bot python -u -m eval.run_matrix_eval \
    "${MATRIX_ARGS[@]}" \
    -- \
    --dataset "eval/datasets/versions/${DATASET_VERSION}/regression.jsonl" \
    --runs-per-case "${RUNS_PER_CASE}" \
    --include-trace-summary

MANIFEST_PATH="$(ls -t "${REPO_ROOT}/eval/reports/${RUN_ID}/matrix/"*_manifest.json | head -n1 || true)"
if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "[SkillMatrix][Error] matrix manifest not found under eval/reports/${RUN_ID}/matrix" >&2
  exit 1
fi
MANIFEST_REL="eval/reports/${RUN_ID}/matrix/$(basename "${MANIFEST_PATH}")"

# Step 4: v1 leaderboard.
compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval/reports:/app/eval/reports" \
  bot python -u -m eval.build_task_eval_v1_leaderboard \
    --manifest "${MANIFEST_REL}" \
    --baseline-group "${BASELINE_GROUP}" \
    --output-json "eval/reports/${RUN_ID}/leaderboard/latest.json" \
    --output-md "eval/reports/${RUN_ID}/leaderboard/latest.md"

echo "[SkillMatrix] done"
echo "[SkillMatrix] run_id=${RUN_ID}"
echo "[SkillMatrix] dataset_version=${DATASET_VERSION}"
echo "[SkillMatrix] matrix_manifest=${MANIFEST_REL}"
echo "[SkillMatrix] leaderboard=eval/reports/${RUN_ID}/leaderboard/latest.json"
