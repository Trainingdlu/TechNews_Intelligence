# TechNews Project Structure (2026-04, Unified ReAct Runtime)

This document describes the current `main` branch layout and ownership.

## 1) Repository root

| Path | Purpose |
| --- | --- |
| `.github/workflows/` | CI and manual evaluation workflows |
| `agents/` | Agent service code (runtime, tools, tests) |
| `deployment/` | Docker Compose and deployment configs |
| `docs/` | Project and testing documentation |
| `etl_workflow/` | n8n workflow JSON files |
| `frontend/` | Web frontend code |
| `sql/` | Database schema and analytics SQL |
| `assets/` | Screenshots and static assets |

## 2) `agents/` package

### 2.1 Runtime entrypoints

| File | Purpose |
| --- | --- |
| `agents/agent.py` | Main runtime (ReAct loop, tools, evidence gate, post-processing) |
| `agents/api.py` | FastAPI entrypoint (web) |
| `agents/bot.py` | Telegram bot entrypoint |
| `agents/cli.py` | Local CLI entrypoint |
| `agents/prompts.py` | System prompts and output constraints |
| `agents/tools.py` | Retrieval and analysis tool implementations |
| `agents/db.py` | PostgreSQL pool and DB helpers |
| `agents/mail.py` | Notification mail helpers |

### 2.2 Shared runtime components (`agents/core/`)

| File | Purpose |
| --- | --- |
| `skill_contracts.py` | `SkillEnvelope` and output contract |
| `skill_registry.py` | Typed skill registration and dispatch |
| `tool_hooks.py` | Pre/post hook validation and auditing |
| `runtime_factories.py` | Default builders for registry and hook runner |
| `evidence.py` | URL extraction, citation normalization, sources section |
| `metrics.py` | In-memory route metrics for ReAct runtime |
| `role_policy.py` | Role/skill allowlist for extension points |

### 2.3 Graph compatibility layer (`agents/graph/`)

| File | Purpose |
| --- | --- |
| `workflow.py` | Compatibility shim that re-exports runtime factories |
| `__init__.py` | Compatibility exports |

### 2.4 MCP extension layer (`agents/mcp/`)

| File | Purpose |
| --- | --- |
| `client.py` | MCP client (local/stdio, namespaced tool routing) |
| `server.py` | In-process MCP server for newsdb tools |
| `stdio_server.py` | Standalone stdio MCP server entrypoint |

Note: MCP is preserved as an extension layer. The default online request path remains ReAct + local skill dispatch in `agents/agent.py`.

## 3) Tests

| Path | Purpose |
| --- | --- |
| `agents/tests/unit/` | Unit tests for runtime, tools, MCP, hooks, metrics |
| `docs/testing/` | Testing and evaluation documentation |

## 4) Runtime summary

1. `main` uses a single ReAct runtime path.
2. Skill registry, hooks, and evidence processing are unified in that path.
3. `agents/graph/` is a compatibility namespace, not an active DAG orchestrator.
4. MCP remains optional and is not a required default dependency for each request.
