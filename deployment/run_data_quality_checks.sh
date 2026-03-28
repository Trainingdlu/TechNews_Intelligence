#!/usr/bin/env bash
set -euo pipefail

# Read-only data quality checks for TechNews.
# Usage:
#   bash deployment/run_data_quality_checks.sh
#   bash deployment/run_data_quality_checks.sh 48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deployment"
CHECK_SQL="${REPO_ROOT}/sql/infrastructure/data_quality_checks.sql"

CHECK_HOURS="${1:-24}"

if ! [[ "${CHECK_HOURS}" =~ ^[0-9]+$ ]]; then
  echo "check_hours must be a positive integer, got: ${CHECK_HOURS}" >&2
  exit 1
fi

if [[ ! -f "${CHECK_SQL}" ]]; then
  echo "Missing SQL file: ${CHECK_SQL}" >&2
  exit 1
fi

cd "${DEPLOY_DIR}"

if [[ ! -f ".env" ]]; then
  echo "Missing deployment/.env. Please copy from .env.example and fill DB credentials first." >&2
  exit 1
fi

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

echo "Running read-only data quality checks (window=${CHECK_HOURS}h)..."
cat "${CHECK_SQL}" | "${COMPOSE_CMD[@]}" exec -T postgres \
  psql -v ON_ERROR_STOP=1 -v check_hours="${CHECK_HOURS}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"

echo "Done."
