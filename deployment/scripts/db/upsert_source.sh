#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMMON_LIB="${SCRIPT_DIR}/common.sh"
FRAMEWORK_SCRIPT="${SCRIPT_DIR}/apply_source_framework_migration.sh"

SOURCE_KEY=""
SOURCE_NAME=""
SOURCE_PLATFORM="DirectRSS"
SIGNAL_ORIGIN="Official"
FETCH_TYPE="rss"
ENDPOINT=""
IS_ACTIVE="true"
PRIORITY="100"
EXTRA_CONFIG="{}"
SKIP_MIGRATION=false
APPLY_DEFAULT_SEEDS=false

usage() {
  cat <<'EOF'
Usage:
  bash deployment/scripts/db/upsert_source.sh --source-key <key> --source-name <name> --endpoint <url> [options]

Required:
  --source-key <value>        Unique source key, e.g. openai_blog
  --source-name <value>       Display name, e.g. OpenAI Blog
  --endpoint <value>          RSS/API endpoint

Optional:
  --source-platform <value>   Default: DirectRSS
  --signal-origin <value>     Default: Official (Official/Academic/Media/Community/Other)
  --fetch-type <value>        Default: rss
  --priority <int>            Default: 100 (smaller means earlier)
  --is-active <bool>          true/false, default true
  --extra-config <json>       JSON string, default {}
  --skip-migration            Do not run apply_source_framework_migration.sh first
  --apply-default-seeds       When migration runs, also apply seed_source_*.sql bundle
  -h, --help                  Show help
EOF
}

if [[ ! -f "${COMMON_LIB}" ]]; then
  echo "Missing common helper: ${COMMON_LIB}" >&2
  exit 1
fi
source "${COMMON_LIB}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-key)
      [[ $# -ge 2 ]] || { echo "Missing value for --source-key" >&2; exit 1; }
      SOURCE_KEY="$2"
      shift 2
      ;;
    --source-name)
      [[ $# -ge 2 ]] || { echo "Missing value for --source-name" >&2; exit 1; }
      SOURCE_NAME="$2"
      shift 2
      ;;
    --source-platform)
      [[ $# -ge 2 ]] || { echo "Missing value for --source-platform" >&2; exit 1; }
      SOURCE_PLATFORM="$2"
      shift 2
      ;;
    --signal-origin)
      [[ $# -ge 2 ]] || { echo "Missing value for --signal-origin" >&2; exit 1; }
      SIGNAL_ORIGIN="$2"
      shift 2
      ;;
    --fetch-type)
      [[ $# -ge 2 ]] || { echo "Missing value for --fetch-type" >&2; exit 1; }
      FETCH_TYPE="$2"
      shift 2
      ;;
    --endpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --endpoint" >&2; exit 1; }
      ENDPOINT="$2"
      shift 2
      ;;
    --priority)
      [[ $# -ge 2 ]] || { echo "Missing value for --priority" >&2; exit 1; }
      PRIORITY="$2"
      shift 2
      ;;
    --is-active)
      [[ $# -ge 2 ]] || { echo "Missing value for --is-active" >&2; exit 1; }
      IS_ACTIVE="$2"
      shift 2
      ;;
    --extra-config)
      [[ $# -ge 2 ]] || { echo "Missing value for --extra-config" >&2; exit 1; }
      EXTRA_CONFIG="$2"
      shift 2
      ;;
    --skip-migration)
      SKIP_MIGRATION=true
      shift
      ;;
    --apply-default-seeds)
      APPLY_DEFAULT_SEEDS=true
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

if [[ -z "${SOURCE_KEY}" || -z "${SOURCE_NAME}" || -z "${ENDPOINT}" ]]; then
  echo "Missing required args. --source-key/--source-name/--endpoint are required." >&2
  usage >&2
  exit 1
fi

if ! [[ "${SOURCE_KEY}" =~ ^[a-z0-9][a-z0-9_]{1,98}$ ]]; then
  echo "--source-key must match ^[a-z0-9][a-z0-9_]{1,98}$, got: ${SOURCE_KEY}" >&2
  exit 1
fi

if ! [[ "${ENDPOINT}" =~ ^https?:// ]]; then
  echo "--endpoint must start with http:// or https://, got: ${ENDPOINT}" >&2
  exit 1
fi

if ! [[ "${PRIORITY}" =~ ^-?[0-9]+$ ]]; then
  echo "--priority must be an integer, got: ${PRIORITY}" >&2
  exit 1
fi

IS_ACTIVE="$(echo "${IS_ACTIVE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${IS_ACTIVE}" != "true" && "${IS_ACTIVE}" != "false" ]]; then
  echo "--is-active must be true or false, got: ${IS_ACTIVE}" >&2
  exit 1
fi

if [[ "${SKIP_MIGRATION}" == false ]]; then
  if [[ ! -f "${FRAMEWORK_SCRIPT}" ]]; then
    echo "Missing migration runner: ${FRAMEWORK_SCRIPT}" >&2
    exit 1
  fi
  echo "Ensuring source framework schema/view..."
  if [[ "${APPLY_DEFAULT_SEEDS}" == true ]]; then
    bash "${FRAMEWORK_SCRIPT}"
  else
    bash "${FRAMEWORK_SCRIPT}" --skip-seeds
  fi
fi

db_init_runtime "${REPO_ROOT}"
db_ensure_postgres_running

echo "Upserting source into source_registry: ${SOURCE_KEY}"
db_psql_sql "
INSERT INTO public.source_registry (
    source_key,
    source_name,
    source_platform,
    signal_origin,
    fetch_type,
    endpoint,
    is_active,
    priority,
    extra_config
)
VALUES (
    :'source_key',
    :'source_name',
    :'source_platform',
    :'signal_origin',
    :'fetch_type',
    :'endpoint',
    (:'is_active')::boolean,
    (:'priority')::integer,
    (:'extra_config')::jsonb
)
ON CONFLICT (source_key) DO UPDATE
SET
    source_name = EXCLUDED.source_name,
    source_platform = EXCLUDED.source_platform,
    signal_origin = EXCLUDED.signal_origin,
    fetch_type = EXCLUDED.fetch_type,
    endpoint = EXCLUDED.endpoint,
    is_active = EXCLUDED.is_active,
    priority = EXCLUDED.priority,
    extra_config = EXCLUDED.extra_config;
" \
  -v source_key="${SOURCE_KEY}" \
  -v source_name="${SOURCE_NAME}" \
  -v source_platform="${SOURCE_PLATFORM}" \
  -v signal_origin="${SIGNAL_ORIGIN}" \
  -v fetch_type="${FETCH_TYPE}" \
  -v endpoint="${ENDPOINT}" \
  -v is_active="${IS_ACTIVE}" \
  -v priority="${PRIORITY}" \
  -v extra_config="${EXTRA_CONFIG}"

echo "Done. Source upserted successfully."
