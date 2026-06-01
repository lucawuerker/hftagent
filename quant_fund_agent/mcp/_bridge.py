"""Generic synchronous bridge to the quant fund's MCP stdio servers.

The agents run inside synchronous LangGraph nodes, but the MCP SDK is async-only
and each stdio session must stay open across many calls (so a server loads its
heavy data — e.g. the factor panel — just once and reuses it).  This module
bridges the two: it runs a single persistent asyncio event loop on a background
thread and, for each server module, connects to one long-lived server subprocess
and exposes a plain synchronous :meth:`MCPBridge.call`.

One :class:`MCPBridge` instance exists per server module (keyed in a registry),
so ``quant-modeling``, ``quant-research`` etc. each get their own subprocess and
cache while sharing the single background event loop.

Each domain facade (``client.py``, ``catalog_client.py`` …) decides — via
:func:`use_mcp` — whether to go over MCP or call the underlying service module
in-process.  The in-process path computes byte-for-byte identical results because
both paths delegate to the same ``*_service`` functions.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

log = logging.getLogger("mcp.bridge")

_CALL_TIMEOUT_S = float(os.getenv("QF_MCP_TIMEOUT", os.getenv("MODELING_MCP_TIMEOUT", "900")))


def use_mcp(server_env: str) -> bool:
    """Whether ``server_env`` should go over MCP.

    Resolution order: the per-server override (e.g. ``RESEARCH_USE_MCP``) wins if
    set; otherwise the global ``QF_USE_MCP`` (default ``1``) applies.  ``0`` /
    ``false`` / ``no`` disable MCP and select the in-process fallback.
    """
    raw = os.getenv(server_env)
    if raw is None:
        raw = os.getenv("QF_USE_MCP", "1")
    return raw.strip().lower() not in ("0", "false", "no")


def _result_text(result: Any) -> str:
    parts: list[str] = []
    for chunk in getattr(result, "content", None) or []:
        text = getattr(chunk, "text", None)
        if text is not None:
            parts.append(text)
    return "".join(parts)


def _is_connection_dead(exc: BaseException) -> bool:
    """Whether ``exc`` signals the stdio session/subprocess has gone away.

    A long ``fit_and_backtest`` can outlive (or crash) the server subprocess; the
    next call then hits a closed stream.  We treat these as recoverable and
    reconnect rather than letting them crash the pipeline.
    """
    name = type(exc).__name__
    if name in ("ClosedResourceError", "BrokenResourceError", "EndOfStream"):
        return True
    msg = str(exc).lower()
    return "connection closed" in msg or "broken" in msg or "closed resource" in msg


# ---------------------------------------------------------------------------
# Shared background event loop (one thread for every server's session)
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _shared_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, daemon=True, name="mcp-bridge-loop",
            ).start()
        return _loop


class MCPBridge:
    """A persistent stdio session to a single MCP server module."""

    _instances: "dict[str, MCPBridge]" = {}
    _lock = threading.Lock()

    def __init__(self, server_module: str) -> None:
        self._server_module = server_module
        self._loop = _shared_loop()
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._submit(self._connect()).result(timeout=_CALL_TIMEOUT_S)
        atexit.register(self.close)

    @classmethod
    def instance(cls, server_module: str) -> "MCPBridge":
        with cls._lock:
            bridge = cls._instances.get(server_module)
            if bridge is None:
                bridge = cls(server_module)
                cls._instances[server_module] = bridge
            return bridge

    def _submit(self, coro: Any):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self._server_module],
            env=os.environ.copy(),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        log.info("connected to MCP server %s", self._server_module)

    async def _reconnect(self) -> None:
        """Tear down the (likely broken) session and spin up a fresh subprocess."""
        old, self._stack, self._session = self._stack, None, None
        if old is not None:
            try:
                await old.aclose()
            except Exception:  # the resource is already broken — ignore
                pass
        await self._connect()

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._session.call_tool(name, arguments=arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool {name!r} errored: {_result_text(result)}")
        return json.loads(_result_text(result))

    async def _call_with_reconnect(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await self._call(name, arguments)
        except BaseException as e:
            if not _is_connection_dead(e):
                raise
            # Heavy tools (e.g. fit_and_backtest / run_tests) can themselves crash
            # the server — usually by OOM while loading the full panel.  Blindly
            # retrying re-runs that expensive, likely-to-crash-again work.  So we
            # reconnect (to keep the *next* call alive) but only auto-retry cheap,
            # idempotent read calls; for the rest we re-raise and let the caller
            # decide (the agents record the failure and move on).
            retryable = name.startswith("list")
            log.warning(
                "MCP session to %s died (%s) — reconnecting%s.",
                self._server_module, type(e).__name__,
                f" and retrying {name!r}" if retryable else f"; not retrying {name!r}",
            )
            await self._reconnect()
            if retryable:
                return await self._call(name, arguments)
            raise

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._submit(
            self._call_with_reconnect(name, arguments)
        ).result(timeout=_CALL_TIMEOUT_S)

    def close(self) -> None:
        if self._stack is None:
            return
        try:
            self._submit(self._stack.aclose()).result(timeout=10)
        except Exception:  # best-effort shutdown
            pass
        finally:
            self._stack = None
