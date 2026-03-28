#!/usr/bin/env bash
set -euo pipefail

# Idempotent migration runner for source framework.
# Reusable for schema/table extension + source seed bundle.
# Usage:
#   bash deployment/apply_source_framework_migration.sh
#   bash deployment/apply_source_framework_migration.sh --skip-seeds
#   bash deployment/apply_source_framework_migration.sh --seed-file sql/infrastructure/seed_source_xxx.sql

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_LIB="${SCRIPT_DIR}/db_common.sh"
MIGRATION_SQL="${REPO_ROOT}/sql/infrastructure/source_framework_migration.sql"
VIEW_SQL="${REPO_ROOT}/sql/infrastructure/view_logic.sql"
SEED_GLOB_DIR="${REPO_ROOT}/sql/infrastructure"
SKIP_SEEDS=false
SEED_FILES=()

usage() {
  cat <<'EOF'
Usage:
  bash deployment/apply_source_framework_migration.sh [options]

Options:
  --skip-seeds             Apply schema/view only, do not apply seed_source_*.sql.
  --seed-file <path>       Apply specific seed SQL file (can be repeated).
                           If omitted, all sql/infrastructure/seed_source_*.sql are applied.
  -h, --help               Show this help message.
EOF
}

if [[ ! -f "${COMMON_LIB}" ]]; then
  echo "Missing common helper: ${COMMON_LIB}" >&2
  exit 1
fi
source "${COMMON_LIB}"

resolve_seed_file() {
  local raw="$1"
  local resolved=""
  if [[ "${raw}" = /* ]]; then
    resolved="${raw}"
  elif [[ -f "${raw}" ]]; then
    resolved="$(cd "$(dirname "${raw}")" && pwd)/$(basename "${raw}")"
  else
    resolved="${REPO_ROOT}/${raw}"
  fi
  printf '%s' "${resolved}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-seeds)
      SKIP_SEEDS=true
      shift
      ;;
    --seed-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --seed-file" >&2; exit 1; }
      SEED_FILES+=("$(resolve_seed_file "$2")")
      shift 2
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

db_require_file "${MIGRATION_SQL}" "migration SQL"
db_require_file "${VIEW_SQL}" "view SQL"

if [[ "${SKIP_SEEDS}" == false && "${#SEED_FILES[@]}" -eq 0 ]]; then
  while IFS= read -r file; do
    SEED_FILES+=("${file}")
  done < <(find "${SEED_GLOB_DIR}" -maxdepth 1 -type f -name 'seed_source_*.sql' | sort)
fi

db_init_runtime "${REPO_ROOT}"
db_ensure_postgres_running

echo "Applying source framework migration..."
db_psql_file "${MIGRATION_SQL}"

echo "Refreshing dashboard view logic..."
db_psql_file "${VIEW_SQL}"

if [[ "${SKIP_SEEDS}" == true ]]; then
  echo "Skipping seed SQL as requested (--skip-seeds)."
elif [[ "${#SEED_FILES[@]}" -eq 0 ]]; then
  echo "No seed_source_*.sql found, skipping source seed step."
else
  echo "Applying source seed files..."
  for seed_file in "${SEED_FILES[@]}"; do
    db_require_file "${seed_file}" "source seed SQL"
    echo " - $(basename "${seed_file}")"
    db_psql_file "${seed_file}"
  done
fi

echo "Done. Source framework migration applied successfully."
