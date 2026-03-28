#!/usr/bin/env bash

# Common helpers for deployment DB scripts.
# shellcheck shell=bash

DB_RUNTIME_READY=0
POSTGRES_USER="${POSTGRES_USER:-}"
POSTGRES_DB="${POSTGRES_DB:-}"
DEPLOY_DIR=""
REPO_ROOT=""
COMPOSE_CMD=()

db_die() {
  echo "$*" >&2
  exit 1
}

db_read_env_var() {
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

db_detect_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
  else
    db_die "Neither 'docker compose' nor 'docker-compose' is available."
  fi
}

db_require_file() {
  local file="$1"
  local label="${2:-SQL file}"
  if [[ ! -f "${file}" ]]; then
    db_die "Missing ${label}: ${file}"
  fi
}

db_init_runtime() {
  local repo_root="$1"
  REPO_ROOT="${repo_root}"
  DEPLOY_DIR="${REPO_ROOT}/deployment"

  if [[ ! -f "${DEPLOY_DIR}/.env" ]]; then
    db_die "Missing deployment/.env. Please copy from .env.example and fill DB credentials first."
  fi

  if [[ -z "${POSTGRES_USER:-}" ]]; then
    POSTGRES_USER="$(db_read_env_var POSTGRES_USER "${DEPLOY_DIR}/.env")"
  fi
  if [[ -z "${POSTGRES_DB:-}" ]]; then
    POSTGRES_DB="$(db_read_env_var POSTGRES_DB "${DEPLOY_DIR}/.env")"
  fi

  if [[ -z "${POSTGRES_USER:-}" || -z "${POSTGRES_DB:-}" ]]; then
    db_die "POSTGRES_USER / POSTGRES_DB must be set in deployment/.env"
  fi

  db_detect_compose_cmd
  DB_RUNTIME_READY=1
}

db_assert_runtime() {
  if [[ "${DB_RUNTIME_READY}" -ne 1 ]]; then
    db_die "DB runtime is not initialized. Call db_init_runtime first."
  fi
}

db_ensure_postgres_running() {
  db_assert_runtime
  echo "Starting postgres service if needed..."
  (
    cd "${DEPLOY_DIR}"
    "${COMPOSE_CMD[@]}" up -d postgres >/dev/null
  )
}

db_psql_file() {
  db_assert_runtime
  local sql_file="$1"
  shift || true
  db_require_file "${sql_file}" "SQL file"
  (
    cd "${DEPLOY_DIR}"
    cat "${sql_file}" | "${COMPOSE_CMD[@]}" exec -T postgres \
      psql -v ON_ERROR_STOP=1 "$@" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"
  )
}

db_psql_sql() {
  db_assert_runtime
  local sql="$1"
  shift || true
  (
    cd "${DEPLOY_DIR}"
    printf '%s\n' "${sql}" | "${COMPOSE_CMD[@]}" exec -T postgres \
      psql -v ON_ERROR_STOP=1 "$@" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"
  )
}
