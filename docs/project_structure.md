# TechNews Project Structure (2026-04, Unified ReAct Runtime)

This document describes the current `main` branch layout and ownership.

## 1) Repository root

| Path | Purpose |
| --- | --- |
| `.github/workflows/` | CI and manual evaluation workflows |
| `agent/` | Agent runtime core (engine, tools, prompts, skill infra, MCP adapter) |
| `app/` | Runtime entrypoints (`api.py`, `bot.py`, `cli.py`) |
| `services/` | Shared service adapters (`db.py`, `mail.py`) |
| `eval/` | Evaluation engine, datasets, reports |
| `tests/` | Unit tests and test helpers |
| `deployment/` | Docker Compose and deployment configs |
| `docs/` | Project and testing documentation |
| `etl_workflow/` | n8n workflow JSON files |
| `frontend/` | Web frontend code |
| `sql/` | Database schema and analytics SQL |
| `assets/` | Screenshots and static assets |

## 2) Runtime code ownership

### 2.1 Agent core (`agent/`)

| File / Path | Purpose |
| --- | --- |
| `agent/agent.py` | Main runtime (ReAct loop, tools, evidence gate, post-processing) |
| `agent/prompts.py` | System prompts and output constraints |
| `agent/tools.py` | Retrieval and analysis tool implementations |
| `agent/core/` | Skill contracts/registry, tool hooks, metrics, evidence pipeline |
| `agent/mcp/` | MCP client/server/stdio adapters |

### 2.2 Entrypoints (`app/`)

| File | Purpose |
| --- | --- |
| `app/api.py` | FastAPI entrypoint (web) |
| `app/bot.py` | Telegram bot entrypoint |
| `app/cli.py` | Local CLI entrypoint |

### 2.3 Service adapters (`services/`)

| File | Purpose |
| --- | --- |
| `services/db.py` | PostgreSQL pool and DB helpers |
| `services/mail.py` | Notification mail helpers |

## 3) Quality and evaluation

| Path | Purpose |
| --- | --- |
| `eval/` | Eval runner, metric core, dataset loader, datasets, reports |
| `tests/unit/` | Unit tests for runtime, tools, MCP, hooks, and eval components |
| `docs/testing/` | Testing and evaluation documentation |

## 4) Runtime summary

1. `main` uses a single ReAct runtime path.
2. Skill registry, hooks, and evidence processing are unified in that path.
3. MCP is preserved as an extension layer, not the mandatory default request path.
