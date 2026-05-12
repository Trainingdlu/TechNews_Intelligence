#!/usr/bin/env bash
set -euo pipefail

# Explicit seed runner. Keep this out of automatic deploys.
# Usage:
#   bash deployment/scripts/db/apply_seed.sh
#   bash deployment/scripts/db/apply_seed.sh --seed-file sql/infrastructure/seeds/seed_entity_core.sql

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMMON_LIB="${SCRIPT_DIR}/common.sh"
SEED_GLOB_DIR="${REPO_ROOT}/sql/infrastructure/seeds"
SEED_FILES=()

usage() {
  cat <<'EOF'
Usage:
  bash deployment/scripts/db/apply_seed.sh [options]

Options:
  --seed-file <path>       Apply a specific seed SQL file. Can be repeated.
                           If omitted, seed_source_*.sql and seed_entity_*.sql are applied.
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

if [[ "${#SEED_FILES[@]}" -eq 0 ]]; then
  while IFS= read -r file; do
    SEED_FILES+=("${file}")
  done < <(
    find "${SEED_GLOB_DIR}" -maxdepth 1 -type f \
      \( -name 'seed_source_*.sql' -o -name 'seed_entity_*.sql' \) | sort
  )
fi

db_init_runtime "${REPO_ROOT}"
db_ensure_postgres_running

if [[ "${#SEED_FILES[@]}" -eq 0 ]]; then
  echo "No seed SQL found, skipping seed step."
  exit 0
fi

echo "Applying seed files..."
for seed_file in "${SEED_FILES[@]}"; do
  db_require_file "${seed_file}" "seed SQL"
  echo " - $(basename "${seed_file}")"
  db_psql_file "${seed_file}"
done

echo "Done. Seed SQL applied successfully."
