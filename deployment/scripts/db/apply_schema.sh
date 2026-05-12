#!/usr/bin/env bash
set -euo pipefail

# Idempotent schema/view runner for deployment.
# It intentionally does not apply seed SQL; use apply_seed.sh for data seeds.
# Usage:
#   bash deployment/scripts/db/apply_schema.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMMON_LIB="${SCRIPT_DIR}/common.sh"
SCHEMA_SQL="${REPO_ROOT}/sql/infrastructure/schema/schema_ddl.sql"
VIEW_SQL="${REPO_ROOT}/sql/infrastructure/views/view_dashboard_news.sql"

usage() {
  cat <<'EOF'
Usage:
  bash deployment/scripts/db/apply_schema.sh

Applies idempotent database schema DDL and dashboard view SQL.
Seed SQL is deliberately excluded from this deployment path.
EOF
}

if [[ ! -f "${COMMON_LIB}" ]]; then
  echo "Missing common helper: ${COMMON_LIB}" >&2
  exit 1
fi
source "${COMMON_LIB}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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

db_require_file "${SCHEMA_SQL}" "schema DDL SQL"
db_require_file "${VIEW_SQL}" "view SQL"

db_init_runtime "${REPO_ROOT}"
db_ensure_postgres_running

echo "Applying schema DDL..."
db_psql_file "${SCHEMA_SQL}"

echo "Refreshing dashboard view logic..."
db_psql_file "${VIEW_SQL}"

echo "Done. Schema and view SQL applied successfully."
