#!/usr/bin/env bash
set -euo pipefail

# Idempotent migration runner for source framework.
# Usage:
#   bash deployment/apply_source_framework_migration.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deployment"
MIGRATION_SQL="${REPO_ROOT}/sql/infrastructure/source_framework_migration.sql"
VIEW_SQL="${REPO_ROOT}/sql/infrastructure/view_logic.sql"

if [[ ! -f "${MIGRATION_SQL}" ]]; then
  echo "Missing migration SQL: ${MIGRATION_SQL}" >&2
  exit 1
fi

if [[ ! -f "${VIEW_SQL}" ]]; then
  echo "Missing view SQL: ${VIEW_SQL}" >&2
  exit 1
fi

cd "${DEPLOY_DIR}"

if [[ ! -f ".env" ]]; then
  echo "Missing deployment/.env. Please copy from .env.example and fill DB credentials first." >&2
  exit 1
fi

# Read only required DB vars from .env without executing it.
read_env_var() {
  local key="$1"
  local file="$2"
  local raw
  raw="$(grep -E "^${key}=" "${file}" | tail -n 1 | cut -d '=' -f2- || true)"
  raw="${raw%$'\r'}"
  raw="${raw%\"}"
  raw="${raw#\"}"
  raw="${raw%\'}"
  raw="${raw#\'}"
  printf '%s' "${raw}"
}

POSTGRES_USER="${POSTGRES_USER:-$(read_env_var POSTGRES_USER .env)}"
POSTGRES_DB="${POSTGRES_DB:-$(read_env_var POSTGRES_DB .env)}"

if [[ -z "${POSTGRES_USER:-}" || -z "${POSTGRES_DB:-}" ]]; then
  echo "POSTGRES_USER / POSTGRES_DB must be set in deployment/.env" >&2
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

echo "Starting postgres service if needed..."
"${COMPOSE_CMD[@]}" up -d postgres >/dev/null

echo "Applying source framework migration..."
cat "${MIGRATION_SQL}" | "${COMPOSE_CMD[@]}" exec -T postgres \
  psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"

echo "Refreshing dashboard view logic..."
cat "${VIEW_SQL}" | "${COMPOSE_CMD[@]}" exec -T postgres \
  psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"

echo "Done. Source framework migration applied successfully."
