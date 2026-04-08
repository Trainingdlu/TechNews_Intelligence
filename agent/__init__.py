"""TechNews Intelligence agent package exports."""

from services.db import close_db_pool, init_db_pool
from .agent import (
    AgentGenerationError,
    create_agent_chat,
    generate_response,
    generate_response_eval_payload,
    generate_response_payload,
    get_last_tool_calls_snapshot,
    get_route_metrics_snapshot,
    reset_route_metrics,
)

__all__ = [
    "AgentGenerationError",
    "init_db_pool",
    "close_db_pool",
    "create_agent_chat",
    "generate_response",
    "generate_response_eval_payload",
    "generate_response_payload",
    "get_last_tool_calls_snapshot",
    "get_route_metrics_snapshot",
    "reset_route_metrics",
]
