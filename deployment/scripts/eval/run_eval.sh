#!/usr/bin/env bash
set -euo pipefail

# One-click evaluation pipeline:
# 1) Reuse frozen dataset by fingerprint, or build+audit a new dataset.
# 2) Run serial matrix (G0 -> G1 -> G2), fail-fast.
# 3) Build main leaderboard.
# 4) (Default on) Run Judge audit on baseline + best group.
# 5) Build final markdown report.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deployment"
ENV_FILE="${DEPLOY_DIR}/.env"
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.yml"

LOCK_DIR="${REPO_ROOT}/.locks"
LOCK_FILE="${LOCK_DIR}/run_eval.lock"

RUN_ID="${RUN_ID:-eval_$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_VERSION="${DATASET_VERSION:-v_$(date -u +%Y%m%d_%H%M%S)}"

TASK_FILE="${TASK_FILE:-eval/config/tasks_180.json}"
MATRIX_FILE="${MATRIX_FILE:-eval/config/matrix.json}"
GROUPS="${GROUPS:-G0,G1,G2}"
BASELINE_GROUP="${BASELINE_GROUP:-G0}"

PROVIDER="${PROVIDER:-vertex}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
RUNS_PER_CASE="${RUNS_PER_CASE:-1}"
MATRIX_SLEEP_SECONDS="${MATRIX_SLEEP_SECONDS:-20}"

EVAL_RECALL_PROFILE="${EVAL_RECALL_PROFILE:-base}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
ENABLE_JUDGE="${ENABLE_JUDGE:-on}"

JUDGE_SAMPLE_RATIO="${JUDGE_SAMPLE_RATIO:-0.15}"
JUDGE_MAX_CASES_PER_GROUP="${JUDGE_MAX_CASES_PER_GROUP:-24}"
JUDGE_SAMPLE_SEED="${JUDGE_SAMPLE_SEED:-42}"
JUDGE_RUNS_PER_CASE="${JUDGE_RUNS_PER_CASE:-1}"
JUDGE_SLEEP_SECONDS="${JUDGE_SLEEP_SECONDS:-20}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-${PROVIDER}}"
JUDGE_MODEL="${JUDGE_MODEL:-${MODEL}}"

LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-2}"
LLM_BACKOFF_SEC="${LLM_BACKOFF_SEC:-2}"
STEP_MAX_ATTEMPTS="${STEP_MAX_ATTEMPTS:-4}"
STEP_RETRY_BASE_SLEEP_SEC="${STEP_RETRY_BASE_SLEEP_SEC:-30}"
INTER_LLM_CALL_SLEEP_SEC="${INTER_LLM_CALL_SLEEP_SEC:-45}"
INTER_TASK_SLEEP_SEC="${INTER_TASK_SLEEP_SEC:-0}"
POOLS_PER_GENERATION_CALL="${POOLS_PER_GENERATION_CALL:-2}"
REGEN_POOLS_PER_GENERATION_CALL="${REGEN_POOLS_PER_GENERATION_CALL:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-1}"
ENFORCE_SCENARIO_RETRIEVAL_MAP="${ENFORCE_SCENARIO_RETRIEVAL_MAP:-1}"

AUDIT_MAX_REGEN_ROUNDS="${AUDIT_MAX_REGEN_ROUNDS:-3}"
AUDIT_REGEN_MODE="${AUDIT_REGEN_MODE:-failed_only}"
INITIAL_CASES_PER_AUDIT_CALL="${INITIAL_CASES_PER_AUDIT_CALL:-0}"
REGEN_CASES_PER_AUDIT_CALL="${REGEN_CASES_PER_AUDIT_CALL:-1}"

DATASET_REL="eval/datasets/task_eval_${RUN_ID}.jsonl"
DATASET_MANIFEST_REL="eval/datasets/task_eval_${RUN_ID}_manifest.json"
CHECKPOINT_REL="eval/datasets/task_eval_${RUN_ID}.checkpoint.json"
REPORT_ROOT_REL="eval/reports/${RUN_ID}"
MAIN_MATRIX_DIR_REL="${REPORT_ROOT_REL}/matrix"
JUDGE_MATRIX_DIR_REL="${REPORT_ROOT_REL}/judge_matrix"
MAIN_LEADERBOARD_JSON_REL="${REPORT_ROOT_REL}/leaderboard/latest.json"
MAIN_LEADERBOARD_MD_REL="${REPORT_ROOT_REL}/leaderboard/latest.md"
JUDGE_LEADERBOARD_JSON_REL="${REPORT_ROOT_REL}/judge_leaderboard/latest.json"
JUDGE_LEADERBOARD_MD_REL="${REPORT_ROOT_REL}/judge_leaderboard/latest.md"
FINAL_REPORT_MD_REL="${REPORT_ROOT_REL}/final_report.md"
JUDGE_SAMPLE_SUMMARY_REL="${REPORT_ROOT_REL}/judge_sample_summary.json"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[Eval][Error] missing ${ENV_FILE}. Copy deployment/.env.example first." >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/${TASK_FILE}" ]]; then
  echo "[Eval][Error] task file not found: ${TASK_FILE}" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/${MATRIX_FILE}" ]]; then
  echo "[Eval][Error] matrix file not found: ${MATRIX_FILE}" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[Eval][Error] 'flock' is required." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "[Eval][Error] neither docker compose nor docker-compose is available." >&2
  exit 1
fi

mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[Eval][Error] another run is active. lock=${LOCK_FILE}" >&2
  exit 2
fi

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_on() {
  case "$(lower "${1:-}")" in
    1|true|on|yes) return 0 ;;
    *) return 1 ;;
  esac
}

compose() {
  (
    cd "${DEPLOY_DIR}"
    "${COMPOSE_CMD[@]}" --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
  )
}

step() {
  local name="$1"
  shift
  echo "[Eval][Step] ${name} started"
  if "$@"; then
    echo "[Eval][Step] ${name} done"
    return 0
  fi
  local rc=$?
  echo "[Eval][Step] ${name} failed rc=${rc}" >&2
  return "${rc}"
}

retry_step() {
  local name="$1"
  local max_attempts="$2"
  local base_sleep="$3"
  shift 3
  local attempt=1
  while [[ "${attempt}" -le "${max_attempts}" ]]; do
    echo "[Eval][Retry] ${name} attempt=${attempt}/${max_attempts}"
    if "$@"; then
      return 0
    fi
    if [[ "${attempt}" -ge "${max_attempts}" ]]; then
      break
    fi
    local sleep_sec=$(( base_sleep * (2 ** (attempt - 1)) ))
    echo "[Eval][Retry] ${name} sleep=${sleep_sec}s before retry"
    sleep "${sleep_sec}"
    attempt=$((attempt + 1))
  done
  return 1
}

validate_matrix_manifest() {
  local manifest_path="$1"
  local tag="$2"
  MANIFEST_PATH="${manifest_path}" TAG="${tag}" python3 - <<'PY'
import json
import os
import pathlib

tag = os.environ["TAG"]
path = pathlib.Path(os.environ["MANIFEST_PATH"])
if (not path.exists()) or path.stat().st_size <= 0:
    raise SystemExit(f"[Eval][Error] {tag} manifest missing/empty: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
failed = int(payload.get("failed_groups", 0) or 0)
if failed > 0:
    raise SystemExit(f"[Eval][Error] {tag} failed_groups={failed}: {path}")
groups = payload.get("groups", [])
if not isinstance(groups, list) or not groups:
    raise SystemExit(f"[Eval][Error] {tag} groups missing: {path}")
for g in groups:
    if not isinstance(g, dict):
        continue
    if str(g.get("status", "")).strip() != "ok":
        raise SystemExit(f"[Eval][Error] {tag} group status not ok: {g}")
    out = pathlib.Path(str(g.get("output", "")).strip())
    if (not out.exists()) or out.stat().st_size <= 0:
        raise SystemExit(f"[Eval][Error] {tag} output missing/empty: {out}")
print(f"[Eval] {tag} manifest integrity ok: {path}")
PY
}

find_reusable_dataset() {
  local fingerprint="$1"
  FINGERPRINT="${fingerprint}" REPO_ROOT="${REPO_ROOT}" python3 - <<'PY'
import json
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
fingerprint = os.environ["FINGERPRINT"].strip()
versions_dir = repo_root / "eval" / "datasets" / "versions"
if not versions_dir.exists():
    print("")
    raise SystemExit(0)

matches = []
for manifest_path in versions_dir.glob("*/manifest.json"):
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if str(payload.get("dataset_fingerprint", "")).strip() != fingerprint:
        continue
    version = manifest_path.parent.name
    dataset = manifest_path.parent / "regression.jsonl"
    if not dataset.exists():
        continue
    matches.append((manifest_path.stat().st_mtime, version, str(dataset.relative_to(repo_root))))

if not matches:
    print("")
    raise SystemExit(0)
matches.sort(key=lambda x: x[0], reverse=True)
_, version, dataset_rel = matches[0]
print(f"{version}|{dataset_rel}")
PY
}

COMMON_BUILD_ARGS=(
  --task-types "${TASK_FILE}"
  --no-enforce-coverage-policy
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --llm-max-retries "${LLM_MAX_RETRIES}"
  --llm-backoff-sec "${LLM_BACKOFF_SEC}"
  --audit-max-regen-rounds "${AUDIT_MAX_REGEN_ROUNDS}"
  --audit-regen-mode "${AUDIT_REGEN_MODE}"
  --initial-cases-per-audit-call "${INITIAL_CASES_PER_AUDIT_CALL}"
  --regen-cases-per-audit-call "${REGEN_CASES_PER_AUDIT_CALL}"
  --inter-llm-call-sleep-sec "${INTER_LLM_CALL_SLEEP_SEC}"
  --inter-task-sleep-sec "${INTER_TASK_SLEEP_SEC}"
)
if is_on "${ENFORCE_SCENARIO_RETRIEVAL_MAP}"; then
  COMMON_BUILD_ARGS+=(--enforce-scenario-retrieval-map)
else
  COMMON_BUILD_ARGS+=(--no-enforce-scenario-retrieval-map)
fi

echo "[Eval] run_id=${RUN_ID}"
echo "[Eval] dataset_version=${DATASET_VERSION}"
echo "[Eval] task_file=${TASK_FILE}"
echo "[Eval] matrix_file=${MATRIX_FILE}"
echo "[Eval] groups=${GROUPS}"
echo "[Eval] provider=${PROVIDER} model=${MODEL}"
echo "[Eval] enable_judge=${ENABLE_JUDGE}"

step "postgres_up" compose up -d postgres

FINGERPRINT_RAW="$(
  compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -e EVAL_RECALL_PROFILE="${EVAL_RECALL_PROFILE}" \
    -v "${REPO_ROOT}/eval:/app/eval" \
    bot python -u -m eval.build_task_dataset \
      "${COMMON_BUILD_ARGS[@]}" \
      --print-fingerprint-only
)"

DATASET_FINGERPRINT="$(
  RAW="${FINGERPRINT_RAW}" python3 - <<'PY'
import json
import os

raw = os.environ.get("RAW", "")
start = raw.find("{")
end = raw.rfind("}")
if start < 0 or end < 0 or end <= start:
    raise SystemExit("fingerprint json not found")
payload = json.loads(raw[start:end + 1])
print(str(payload.get("dataset_fingerprint", "")).strip())
PY
)"
if [[ -z "${DATASET_FINGERPRINT}" ]]; then
  echo "[Eval][Error] failed to resolve dataset fingerprint." >&2
  exit 3
fi
echo "[Eval] dataset_fingerprint=${DATASET_FINGERPRINT}"

FROZEN_DATASET_REL=""
if ! is_on "${FORCE_REBUILD}"; then
  REUSE_HIT="$(find_reusable_dataset "${DATASET_FINGERPRINT}")"
  if [[ -n "${REUSE_HIT}" ]]; then
    DATASET_VERSION="${REUSE_HIT%%|*}"
    FROZEN_DATASET_REL="${REUSE_HIT##*|}"
    echo "[Eval] dataset reuse hit: version=${DATASET_VERSION} dataset=${FROZEN_DATASET_REL}"
  fi
fi

run_build_dataset_with_chunk() {
  local chunk_size="$1"
  local build_args=(
    "${COMMON_BUILD_ARGS[@]}"
    --output "${DATASET_REL}"
    --manifest-output "${DATASET_MANIFEST_REL}"
    --checkpoint-path "${CHECKPOINT_REL}"
    --pools-per-generation-call "${chunk_size}"
    --regen-pools-per-generation-call "${REGEN_POOLS_PER_GENERATION_CALL}"
  )
  if is_on "${RESUME_FROM_CHECKPOINT}"; then
    build_args+=(--resume-from-checkpoint)
  else
    build_args+=(--no-resume-from-checkpoint)
  fi
  retry_step "build_task_dataset(chunk=${chunk_size})" "${STEP_MAX_ATTEMPTS}" "${STEP_RETRY_BASE_SLEEP_SEC}" \
    compose run --rm --no-deps \
      -e PYTHONUNBUFFERED=1 \
      -e EVAL_RECALL_PROFILE="${EVAL_RECALL_PROFILE}" \
      -v "${REPO_ROOT}/eval:/app/eval" \
      bot python -u -m eval.build_task_dataset \
        "${build_args[@]}"
}

if [[ -z "${FROZEN_DATASET_REL}" ]]; then
  if ! run_build_dataset_with_chunk "${POOLS_PER_GENERATION_CALL}"; then
    if [[ "${POOLS_PER_GENERATION_CALL}" -gt 1 ]]; then
      echo "[Eval][Fallback] build failed with pools_per_generation_call=${POOLS_PER_GENERATION_CALL}, retry with 1"
      run_build_dataset_with_chunk "1"
    else
      echo "[Eval][Error] dataset build failed and no fallback available." >&2
      exit 4
    fi
  fi

  if [[ ! -s "${REPO_ROOT}/${DATASET_REL}" ]]; then
    echo "[Eval][Error] dataset missing: ${DATASET_REL}" >&2
    exit 4
  fi
  if [[ ! -s "${REPO_ROOT}/${DATASET_MANIFEST_REL}" ]]; then
    echo "[Eval][Error] dataset manifest missing: ${DATASET_MANIFEST_REL}" >&2
    exit 4
  fi

  step "freeze_dataset_version" compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${REPO_ROOT}/eval/datasets:/app/eval/datasets" \
    bot sh -lc "
      set -euo pipefail
      V='${DATASET_VERSION}'
      mkdir -p /app/eval/datasets/versions/\"\${V}\"
      cp /app/${DATASET_REL} /app/eval/datasets/versions/\"\${V}\"/regression.jsonl
      cp /app/${DATASET_MANIFEST_REL} /app/eval/datasets/versions/\"\${V}\"/manifest.json
    "
  FROZEN_DATASET_REL="eval/datasets/versions/${DATASET_VERSION}/regression.jsonl"
fi

if [[ ! -s "${REPO_ROOT}/${FROZEN_DATASET_REL}" ]]; then
  echo "[Eval][Error] frozen dataset missing: ${FROZEN_DATASET_REL}" >&2
  exit 5
fi
echo "[Eval] frozen_dataset=${FROZEN_DATASET_REL}"

retry_step "run_main_matrix" "${STEP_MAX_ATTEMPTS}" "${STEP_RETRY_BASE_SLEEP_SEC}" \
  compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -e EVAL_RECALL_PROFILE="${EVAL_RECALL_PROFILE}" \
    -e AGENT_MODEL_PROVIDER="${PROVIDER}" \
    -e GEMINI_MODEL="${MODEL}" \
    -e VERTEX_GENERATION_MODEL="${MODEL}" \
    -e VERTEX_MODEL="${MODEL}" \
    -v "${REPO_ROOT}/eval:/app/eval" \
    bot python -u -m eval.run_matrix_eval \
      --matrix "${MATRIX_FILE}" \
      --groups "${GROUPS}" \
      --output-dir "${MAIN_MATRIX_DIR_REL}" \
      -- \
      --dataset "${FROZEN_DATASET_REL}" \
      --runs-per-case "${RUNS_PER_CASE}" \
      --sleep-seconds "${MATRIX_SLEEP_SECONDS}" \
      --include-trace-summary

MAIN_MANIFEST_PATH="$(ls -t "${REPO_ROOT}/${MAIN_MATRIX_DIR_REL}/"*_manifest.json 2>/dev/null | head -n1 || true)"
if [[ -z "${MAIN_MANIFEST_PATH}" ]]; then
  echo "[Eval][Error] main matrix manifest missing: ${MAIN_MATRIX_DIR_REL}" >&2
  exit 6
fi
validate_matrix_manifest "${MAIN_MANIFEST_PATH}" "main"
MAIN_MANIFEST_REL="${MAIN_MATRIX_DIR_REL}/$(basename "${MAIN_MANIFEST_PATH}")"

step "build_main_leaderboard" compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval:/app/eval" \
  bot python -u -m eval.build_task_eval_leaderboard \
    --manifest "${MAIN_MANIFEST_REL}" \
    --baseline-group "${BASELINE_GROUP}" \
    --output-json "${MAIN_LEADERBOARD_JSON_REL}" \
    --output-md "${MAIN_LEADERBOARD_MD_REL}"

if [[ ! -s "${REPO_ROOT}/${MAIN_LEADERBOARD_JSON_REL}" ]]; then
  echo "[Eval][Error] main leaderboard json missing: ${MAIN_LEADERBOARD_JSON_REL}" >&2
  exit 7
fi
if [[ ! -s "${REPO_ROOT}/${MAIN_LEADERBOARD_MD_REL}" ]]; then
  echo "[Eval][Error] main leaderboard md missing: ${MAIN_LEADERBOARD_MD_REL}" >&2
  exit 7
fi

BEST_GROUP="$(
  MAIN_LEADERBOARD="${REPO_ROOT}/${MAIN_LEADERBOARD_JSON_REL}" BASELINE_GROUP="${BASELINE_GROUP}" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MAIN_LEADERBOARD"])
baseline = str(os.environ["BASELINE_GROUP"]).strip()
data = json.loads(path.read_text(encoding="utf-8"))
best_group = baseline
best_score = float("-inf")
for group in data.get("groups", []):
    if not isinstance(group, dict):
        continue
    gid = str(group.get("group_id", "")).strip()
    if not gid or gid == baseline:
        continue
    metrics = group.get("metrics", {})
    if not isinstance(metrics, dict):
        continue
    retrieval = metrics.get("retrieval_rcs", {})
    if not isinstance(retrieval, dict):
        continue
    score = retrieval.get("improvement_score")
    try:
        score_num = float(score)
    except Exception:
        continue
    if score_num > best_score:
        best_score = score_num
        best_group = gid
print(best_group)
PY
)"
if [[ -z "${BEST_GROUP}" ]]; then
  BEST_GROUP="${BASELINE_GROUP}"
fi

JUDGE_LEADERBOARD_JSON_ARG=()
if is_on "${ENABLE_JUDGE}"; then
  JUDGE_SAMPLE_REL="eval/datasets/versions/${DATASET_VERSION}/judge_sample.jsonl"
  step "build_judge_sample" compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${REPO_ROOT}/eval:/app/eval" \
    bot python -u -m eval.sample_task_eval_dataset \
      --dataset "${FROZEN_DATASET_REL}" \
      --output "${JUDGE_SAMPLE_REL}" \
      --sample-ratio "${JUDGE_SAMPLE_RATIO}" \
      --max-cases "${JUDGE_MAX_CASES_PER_GROUP}" \
      --seed "${JUDGE_SAMPLE_SEED}" \
      --summary-output "${JUDGE_SAMPLE_SUMMARY_REL}"

  if [[ ! -s "${REPO_ROOT}/${JUDGE_SAMPLE_REL}" ]]; then
    echo "[Eval][Error] judge sample missing: ${JUDGE_SAMPLE_REL}" >&2
    exit 8
  fi

  if [[ "${BEST_GROUP}" == "${BASELINE_GROUP}" ]]; then
    JUDGE_GROUPS="${BASELINE_GROUP}"
  else
    JUDGE_GROUPS="${BASELINE_GROUP},${BEST_GROUP}"
  fi

  retry_step "run_judge_matrix" "${STEP_MAX_ATTEMPTS}" "${STEP_RETRY_BASE_SLEEP_SEC}" \
    compose run --rm --no-deps \
      -e PYTHONUNBUFFERED=1 \
      -e EVAL_RECALL_PROFILE="${EVAL_RECALL_PROFILE}" \
      -e AGENT_MODEL_PROVIDER="${PROVIDER}" \
      -e GEMINI_MODEL="${MODEL}" \
      -e VERTEX_GENERATION_MODEL="${MODEL}" \
      -e VERTEX_MODEL="${MODEL}" \
      -v "${REPO_ROOT}/eval:/app/eval" \
      bot python -u -m eval.run_matrix_eval \
        --matrix "${MATRIX_FILE}" \
        --groups "${JUDGE_GROUPS}" \
        --output-dir "${JUDGE_MATRIX_DIR_REL}" \
        -- \
        --dataset "${JUDGE_SAMPLE_REL}" \
        --runs-per-case "${JUDGE_RUNS_PER_CASE}" \
        --sleep-seconds "${JUDGE_SLEEP_SECONDS}" \
        --include-trace-summary \
        --enable-llm-judge \
        --judge-provider "${JUDGE_PROVIDER}" \
        --judge-model "${JUDGE_MODEL}"

  JUDGE_MANIFEST_PATH="$(ls -t "${REPO_ROOT}/${JUDGE_MATRIX_DIR_REL}/"*_manifest.json 2>/dev/null | head -n1 || true)"
  if [[ -z "${JUDGE_MANIFEST_PATH}" ]]; then
    echo "[Eval][Error] judge matrix manifest missing: ${JUDGE_MATRIX_DIR_REL}" >&2
    exit 9
  fi
  validate_matrix_manifest "${JUDGE_MANIFEST_PATH}" "judge"
  JUDGE_MANIFEST_REL="${JUDGE_MATRIX_DIR_REL}/$(basename "${JUDGE_MANIFEST_PATH}")"

  step "build_judge_leaderboard" compose run --rm --no-deps \
    -e PYTHONUNBUFFERED=1 \
    -v "${REPO_ROOT}/eval:/app/eval" \
    bot python -u -m eval.build_task_eval_leaderboard \
      --manifest "${JUDGE_MANIFEST_REL}" \
      --baseline-group "${BASELINE_GROUP}" \
      --output-json "${JUDGE_LEADERBOARD_JSON_REL}" \
      --output-md "${JUDGE_LEADERBOARD_MD_REL}"

  if [[ ! -s "${REPO_ROOT}/${JUDGE_LEADERBOARD_JSON_REL}" ]]; then
    echo "[Eval][Error] judge leaderboard json missing: ${JUDGE_LEADERBOARD_JSON_REL}" >&2
    exit 10
  fi
  JUDGE_LEADERBOARD_JSON_ARG=(--judge-leaderboard-json "${JUDGE_LEADERBOARD_JSON_REL}")
fi

step "build_final_report" compose run --rm --no-deps \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}/eval:/app/eval" \
  bot python -u -m eval.build_report \
    --main-leaderboard-json "${MAIN_LEADERBOARD_JSON_REL}" \
    --main-leaderboard-md "${MAIN_LEADERBOARD_MD_REL}" \
    "${JUDGE_LEADERBOARD_JSON_ARG[@]}" \
    --run-id "${RUN_ID}" \
    --dataset-version "${DATASET_VERSION}" \
    --output-md "${FINAL_REPORT_MD_REL}"

if [[ ! -s "${REPO_ROOT}/${FINAL_REPORT_MD_REL}" ]]; then
  echo "[Eval][Error] final report missing: ${FINAL_REPORT_MD_REL}" >&2
  exit 11
fi

echo "[Eval] done"
echo "[Eval] run_id=${RUN_ID}"
echo "[Eval] dataset_version=${DATASET_VERSION}"
echo "[Eval] dataset_fingerprint=${DATASET_FINGERPRINT}"
echo "[Eval] baseline_group=${BASELINE_GROUP}"
echo "[Eval] best_group=${BEST_GROUP}"
echo "[Eval] main_manifest=${MAIN_MANIFEST_REL}"
if [[ -n "${JUDGE_LEADERBOARD_JSON_ARG[*]}" ]]; then
  echo "[Eval] judge_leaderboard=${JUDGE_LEADERBOARD_JSON_REL}"
fi
echo "[Eval] final_report=${FINAL_REPORT_MD_REL}"
