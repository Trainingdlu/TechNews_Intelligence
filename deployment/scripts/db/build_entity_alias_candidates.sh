#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the offline entity-alias candidate pipeline.
# Recommended scheduling target for n8n Schedule Trigger, cron, or Windows Task
# Scheduler. It does not run on the user-facing request path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMMON_LIB="${SCRIPT_DIR}/common.sh"
FRAMEWORK_SCRIPT="${SCRIPT_DIR}/apply_source_framework_migration.sh"

DAYS="30"
LIMIT="1000"
USE_DEEPSEEK=false
DRY_RUN=false
APPLY_MIGRATION=false
PROMOTE_APPROVED=true
INCLUDE_AUTO_APPROVED_PROMOTION=false

usage() {
  cat <<'EOF'
Usage:
  bash deployment/scripts/db/build_entity_alias_candidates.sh [options]

Options:
  --days <int>          News lookback window. Default: 30.
  --limit <int>         Max recent rows to scan. Default: 1000.
  --use-deepseek        Call DeepSeek for merge/disambiguation decisions.
  --dry-run             Print candidates without writing entity_alias_candidate.
  --apply-migration     Apply schema/view/seeds before extraction.
  --skip-migration      Compatibility no-op; migration is skipped by default.
  --skip-promotion      Do not promote previously approved candidates.
  --include-auto-approved-promotion
                        Also promote auto_approved candidates into entity_alias.
                        By default, only manually approved candidates are promoted.
  -h, --help            Show this help message.

Scheduling:
  This wrapper is the preferred target for n8n/cron/Task Scheduler. For a
  conservative daily run, use:
    bash deployment/scripts/db/build_entity_alias_candidates.sh --days 14 --limit 1000 --use-deepseek
EOF
}

if [[ ! -f "${COMMON_LIB}" ]]; then
  echo "Missing common helper: ${COMMON_LIB}" >&2
  exit 1
fi
source "${COMMON_LIB}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days)
      [[ $# -ge 2 ]] || { echo "Missing value for --days" >&2; exit 1; }
      DAYS="$2"
      shift 2
      ;;
    --limit)
      [[ $# -ge 2 ]] || { echo "Missing value for --limit" >&2; exit 1; }
      LIMIT="$2"
      shift 2
      ;;
    --use-deepseek)
      USE_DEEPSEEK=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --apply-migration)
      APPLY_MIGRATION=true
      shift
      ;;
    --skip-migration)
      APPLY_MIGRATION=false
      shift
      ;;
    --skip-promotion)
      PROMOTE_APPROVED=false
      shift
      ;;
    --include-auto-approved-promotion)
      INCLUDE_AUTO_APPROVED_PROMOTION=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${DAYS}" =~ ^[0-9]+$ ]]; then
  echo "--days must be a positive integer, got: ${DAYS}" >&2
  exit 1
fi
if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "--limit must be a positive integer, got: ${LIMIT}" >&2
  exit 1
fi

if [[ "${APPLY_MIGRATION}" == true ]]; then
  bash "${FRAMEWORK_SCRIPT}"
fi

db_init_runtime "${REPO_ROOT}"
db_ensure_postgres_running

ARGS=(
  "${REPO_ROOT}/eval/build_entity_alias_candidates.py"
  --days "${DAYS}"
  --limit "${LIMIT}"
  --env-file "${REPO_ROOT}/deployment/.env"
)

if [[ "${USE_DEEPSEEK}" == true ]]; then
  ARGS+=(--use-deepseek)
fi
if [[ "${DRY_RUN}" == true ]]; then
  ARGS+=(--dry-run)
fi

python "${ARGS[@]}"

if [[ "${DRY_RUN}" == false && "${PROMOTE_APPROVED}" == true ]]; then
  PROMOTE_ARGS=(
    "${REPO_ROOT}/eval/promote_entity_alias_candidates.py"
    --limit "${LIMIT}"
    --env-file "${REPO_ROOT}/deployment/.env"
  )
  if [[ "${INCLUDE_AUTO_APPROVED_PROMOTION}" == true ]]; then
    PROMOTE_ARGS+=(--include-auto-approved)
  fi
  python "${PROMOTE_ARGS[@]}"
fi
