#!/usr/bin/env bash
set -euo pipefail

# Read-only data quality checks for TechNews.
# Usage:
#   bash deployment/run_data_quality_checks.sh
#   bash deployment/run_data_quality_checks.sh 48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_LIB="${SCRIPT_DIR}/db_common.sh"
CHECK_SQL="${REPO_ROOT}/sql/infrastructure/data_quality_checks.sql"

CHECK_HOURS="${1:-24}"

if ! [[ "${CHECK_HOURS}" =~ ^[0-9]+$ ]] || [[ "${CHECK_HOURS}" -le 0 ]]; then
  echo "check_hours must be a positive integer, got: ${CHECK_HOURS}" >&2
  exit 1
fi

if [[ ! -f "${COMMON_LIB}" ]]; then
  echo "Missing common helper: ${COMMON_LIB}" >&2
  exit 1
fi
source "${COMMON_LIB}"

db_require_file "${CHECK_SQL}" "data-quality SQL"
db_init_runtime "${REPO_ROOT}"
db_ensure_postgres_running

echo "Running read-only data quality checks (window=${CHECK_HOURS}h)..."
db_psql_file "${CHECK_SQL}" -v check_hours="${CHECK_HOURS}"

echo "Done."
