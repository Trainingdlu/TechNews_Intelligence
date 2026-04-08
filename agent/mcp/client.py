"""MCP client with namespaced tool routing (in-process + stdio backends)."""

from __future__ import annotations

import itertools
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ..core.skill_contracts import SkillEnvelope, build_error_envelope
from .server import InProcessMCPServer, build_newsdb_server


@dataclass(frozen=True)
class MCPToolRoute:
    qualified_name: str
    server_name: str
    tool_name: str
    description: str = ""


def qualify_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp__{server_name}__{tool_name}"


class _MCPServerBackend(Protocol):
    server_name: str

    def list_tools(self) -> list[dict[str, Any]]:
        ...

    def call_tool(self, tool_name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        ...


class StdioMCPServer:
    """JSON-RPC stdio backend adapter for MCP tool discovery/calls."""

    def __init__(
        self,
        server_name: str,
        command: list[str],
        cwd: str | None = None,
    ) -> None:
        self.server_name = str(server_name).strip()
        if not self.server_name:
            raise ValueError("server_name is required")
        if not command:
            raise ValueError("command is required")

        self._command = [str(part) for part in command]
        self._cwd = cwd
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._request_ids = itertools.count(1)
        self._stdout_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pending_by_id: dict[Any, dict[str, Any]] = {}
        self._reader_thread: threading.Thread | None = None
        self._rpc_timeout_sec = self._resolve_rpc_timeout()
        self._ensure_started()

    @staticmethod
    def _resolve_rpc_timeout() -> float:
        raw = str(os.getenv("AGENT_MCP_RPC_TIMEOUT_SEC", "20")).strip()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 20.0
        return max(1.0, value)

    def _start_reader_thread_locked(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("stdio process stdout is not ready")

        self._stdout_queue = queue.Queue()
        self._pending_by_id = {}

        def _reader() -> None:
            stdout = proc.stdout
            if stdout is None:
                return
            try:
                for line in stdout:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        message = json.loads(text)
                    except Exception:
                        # Non-JSON line on stdout is ignored to keep transport robust.
                        continue
                    if isinstance(message, dict):
                        self._stdout_queue.put(message)
            finally:
                self._stdout_queue.put({"__eof__": True})

        self._reader_thread = threading.Thread(
            target=_reader,
            name=f"mcp-stdio-reader-{self.server_name}",
            daemon=True,
        )
        self._reader_thread.start()

    def _shutdown_process_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._pending_by_id = {}
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass

    def _ensure_started(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return

            self._proc = subprocess.Popen(
                self._command,
                cwd=self._cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            self._start_reader_thread_locked()

            # Minimal MCP init handshake.
            self._rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "technews-intelligence", "version": "0.1.0"},
                },
            )
            self._rpc("notifications/initialized", {}, expect_response=False)

    def _rpc(self, method: str, params: dict[str, Any], expect_response: bool = True) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise RuntimeError("stdio process is not ready")

            request_id = next(self._request_ids)
            req: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
            if expect_response:
                req["id"] = request_id

            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()

            if not expect_response:
                return {}

            deadline = time.monotonic() + self._rpc_timeout_sec
            while True:
                if request_id in self._pending_by_id:
                    message = self._pending_by_id.pop(request_id)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._shutdown_process_locked()
                        raise RuntimeError(f"rpc timeout waiting for method={method}")
                    try:
                        message = self._stdout_queue.get(timeout=remaining)
                    except queue.Empty:
                        self._shutdown_process_locked()
                        raise RuntimeError(f"rpc timeout waiting for method={method}")

                    if message.get("__eof__"):
                        self._shutdown_process_locked()
                        raise RuntimeError("stdio server closed unexpectedly")

                    msg_id = message.get("id")
                    if msg_id is None:
                        # JSON-RPC notifications are legal; ignore and continue.
                        continue
                    if msg_id != request_id:
                        self._pending_by_id[msg_id] = message
                        continue
                    # Matched response for current request_id.

                if not isinstance(message, dict):
                    raise RuntimeError("invalid jsonrpc response type")

                if "error" in message:
                    error = message.get("error")
                    if isinstance(error, dict):
                        code = error.get("code")
                        detail = error.get("message")
                        raise RuntimeError(f"rpc error {code}: {detail}")
                    raise RuntimeError("rpc error")

                result = message.get("result")
                if isinstance(result, dict):
                    return result
                raise RuntimeError("jsonrpc result is not an object")

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._rpc("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise RuntimeError("tools/list result missing tools list")

        rows: list[dict[str, Any]] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("name") or "").strip()
            if not tool_name:
                continue
            input_schema = item.get("inputSchema")
            rows.append(
                {
                    "server_name": self.server_name,
                    "name": tool_name,
                    "qualified_name": qualify_tool_name(self.server_name, tool_name),
                    "description": str(item.get("description") or ""),
                    "input_schema": input_schema if isinstance(input_schema, dict) else {},
                }
            )
        return rows

    def call_tool(self, tool_name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        request_payload = payload or {}
        normalized_tool = str(tool_name).strip()
        try:
            result = self._rpc(
                "tools/call",
                {"name": normalized_tool, "arguments": request_payload},
            )
        except Exception as exc:  # noqa: BLE001
            return build_error_envelope(
                tool=normalized_tool or str(tool_name),
                request=request_payload,
                error="mcp_client_transport_error",
                diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )

        structured = result.get("structuredContent")
        envelope_payload: dict[str, Any] | None = structured if isinstance(structured, dict) else None

        if envelope_payload is None:
            content = result.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    text = str(first.get("text") or "").strip()
                    if text:
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                envelope_payload = parsed
                        except Exception:
                            envelope_payload = None

        if envelope_payload is None:
            return build_error_envelope(
                tool=normalized_tool or str(tool_name),
                request=request_payload,
                error="mcp_client_protocol_error",
                diagnostics={"result": result},
            )

        try:
            envelope = SkillEnvelope.model_validate(envelope_payload)
        except ValidationError as exc:
            return build_error_envelope(
                tool=normalized_tool or str(tool_name),
                request=request_payload,
                error="mcp_client_protocol_error",
                diagnostics={"validation_errors": exc.errors(), "payload": envelope_payload},
            )

        envelope.tool = normalized_tool or envelope.tool
        if not envelope.request:
            envelope.request = request_payload
        envelope.diagnostics.update(
            {
                "mcp_transport": "stdio",
                "mcp_server": self.server_name,
                "mcp_tool": normalized_tool,
            }
        )
        return envelope

    def close(self) -> None:
        with self._lock:
            self._shutdown_process_locked()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class MCPClient:
    """MCP client for discovery and namespaced call routing."""

    def __init__(self) -> None:
        self._servers: dict[str, _MCPServerBackend] = {}
        self._routes: dict[str, MCPToolRoute] = {}

    def register_server(self, server: _MCPServerBackend) -> None:
        self._servers[server.server_name] = server
        self.refresh_server(server.server_name)

    def refresh_server(self, server_name: str) -> None:
        server = self._servers.get(server_name)
        if server is None:
            return

        stale_keys = [k for k, v in self._routes.items() if v.server_name == server_name]
        for key in stale_keys:
            self._routes.pop(key, None)

        for row in server.list_tools():
            tool_name = str(row.get("name") or "").strip()
            if not tool_name:
                continue
            qualified = str(row.get("qualified_name") or "").strip() or qualify_tool_name(server_name, tool_name)
            self._routes[qualified] = MCPToolRoute(
                qualified_name=qualified,
                server_name=server_name,
                tool_name=tool_name,
                description=str(row.get("description") or ""),
            )

    def list_tools(self) -> list[MCPToolRoute]:
        return [self._routes[k] for k in sorted(self._routes.keys())]

    def call_tool(self, qualified_name: str, payload: dict[str, Any] | None = None) -> SkillEnvelope:
        request_payload = payload or {}
        route = self._routes.get(qualified_name)
        if route is None:
            return build_error_envelope(
                tool=qualified_name,
                request=request_payload,
                error="mcp_unknown_namespaced_tool",
                diagnostics={"available_tools": sorted(self._routes.keys())},
            )

        server = self._servers.get(route.server_name)
        if server is None:
            return build_error_envelope(
                tool=qualified_name,
                request=request_payload,
                error="mcp_server_unavailable",
                diagnostics={"server_name": route.server_name},
            )

        try:
            envelope = server.call_tool(route.tool_name, request_payload)
        except Exception as exc:  # noqa: BLE001
            return build_error_envelope(
                tool=qualified_name,
                request=request_payload,
                error="mcp_server_call_failed",
                diagnostics={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
        envelope.diagnostics.update(
            {
                "mcp_qualified_name": qualified_name,
                "mcp_server": route.server_name,
                "mcp_tool": route.tool_name,
            }
        )
        return envelope


_DEFAULT_CLIENT: MCPClient | None = None


def _default_stdio_command(server_name: str) -> list[str]:
    return [sys.executable, "-m", "agent.mcp.stdio_server", "--server-name", server_name]


def _resolve_stdio_command(server_name: str) -> list[str]:
    raw = str(os.getenv("AGENT_MCP_STDIO_COMMAND", "")).strip()
    if raw:
        return shlex.split(raw, posix=os.name != "nt")
    return _default_stdio_command(server_name)


def _project_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def build_default_mcp_client() -> MCPClient:
    """Build default MCP client with NewsDB server (local by default)."""

    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        return _DEFAULT_CLIENT

    server_name = str(os.getenv("AGENT_MCP_SERVER_NAME", "newsdb")).strip() or "newsdb"
    # Safe default: keep in-process transport unless explicitly opting into stdio.
    mode = str(os.getenv("AGENT_MCP_CLIENT_MODE", "local")).strip().lower()

    client = MCPClient()
    if mode in {"local", "inprocess"}:
        client.register_server(build_newsdb_server(server_name=server_name))
        _DEFAULT_CLIENT = client
        return client

    stdio_cwd = str(os.getenv("AGENT_MCP_STDIO_CWD", "")).strip() or _project_root()
    try:
        stdio_server = StdioMCPServer(
            server_name=server_name,
            command=_resolve_stdio_command(server_name),
            cwd=stdio_cwd,
        )
        client.register_server(stdio_server)
    except Exception:
        # Conservative fallback for environments that cannot spawn stdio server.
        client.register_server(build_newsdb_server(server_name=server_name))

    _DEFAULT_CLIENT = client
    return client
