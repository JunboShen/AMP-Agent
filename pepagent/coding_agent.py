#!/usr/bin/env python3
"""Codex MCP bridge for AutoGen.

This module exposes a single tool function, codex_mcp_run, which launches the
Codex MCP server (stdio transport) and runs a one-shot Codex session.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional
try:
    from exceptiongroup import ExceptionGroup
except Exception:  # pragma: no cover - optional backport
    ExceptionGroup = None

try:
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp import ClientSession
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ModuleNotFoundError(
        "Missing dependency 'mcp'. Install with: pip install mcp"
    ) from exc


@contextmanager
def _suppress_mcp_warnings():
    """Temporarily suppress noisy MCP warnings (codex/event notifications)."""
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    try:
        logging.disable(logging.WARNING)
        yield
    finally:
        logging.disable(logging.NOTSET)
        root_logger.setLevel(prev_level)


def _extract_text(result) -> str:
    if not getattr(result, "content", None):
        return ""
    parts = []
    for item in result.content:
        if getattr(item, "type", None) == "text":
            parts.append(getattr(item, "text", ""))
    return "".join(parts).strip()


def _build_codex_env(drop_override: Optional[bool] = None) -> Dict[str, str]:
    env = os.environ.copy()
    if drop_override is None:
        drop = env.get("CODEX_MCP_DROP_OPENAI_KEY", "").lower() in {"1", "true", "yes"}
    else:
        drop = bool(drop_override)
    if drop:
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_API_BASE", None)
        env.pop("OPENAI_BASE_URL", None)
    return env


async def _run_codex_mcp(
    payload: Dict[str, Any],
    server_cwd: Optional[str] = None,
    drop_openai_key: Optional[bool] = None,
) -> str:
    params = StdioServerParameters(
        command="codex",
        args=["mcp-server"],
        env=_build_codex_env(drop_override=drop_openai_key),
        cwd=server_cwd,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("codex", payload)
            return _extract_text(result)


def codex_mcp_run(
    prompt: str,
    cwd: str = ".",
    sandbox: str = "workspace-write",
    approval_policy: str = "never",
    model: Optional[str] = None,
    profile: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    developer_instructions: Optional[str] = None,
    base_instructions: Optional[str] = None,
    compact_prompt: Optional[str] = None,
    drop_openai_key: Optional[bool] = None,
    timeout_sec: Optional[int] = None,
) -> str:
    """Run a one-shot Codex session via MCP and return a JSON summary string."""
    if not prompt:
        return json.dumps({"status": "error", "error": "prompt is required"})

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "cwd": cwd,
        "sandbox": sandbox,
        "approval-policy": approval_policy,
    }
    if model and "codex" in model:
        payload["model"] = model
    if profile:
        payload["profile"] = profile
    if config:
        payload["config"] = config
    if developer_instructions:
        payload["developer-instructions"] = developer_instructions
    if base_instructions:
        payload["base-instructions"] = base_instructions
    if compact_prompt:
        payload["compact-prompt"] = compact_prompt

    try:
        with _suppress_mcp_warnings():
            timeout_val = timeout_sec
            if timeout_val is None:
                try:
                    timeout_val = int(os.environ.get("CODEX_MCP_TIMEOUT_SEC", "300"))
                except Exception:
                    timeout_val = 300
            coro = _run_codex_mcp(payload, server_cwd=cwd, drop_openai_key=drop_openai_key)
            text = asyncio.run(asyncio.wait_for(coro, timeout=timeout_val))
        return json.dumps({"status": "ok", "content": text})
    except asyncio.TimeoutError:  # pragma: no cover - runtime errors
        return json.dumps({"status": "error", "error": "codex_mcp_run timed out"})
    except Exception as exc:  # pragma: no cover - runtime errors
        msg = str(exc)
        if ExceptionGroup is not None and isinstance(exc, ExceptionGroup):
            try:
                details = "; ".join(str(e) for e in exc.exceptions)
                if details:
                    msg = f"{msg} | details: {details}"
            except Exception:
                pass
        if "api.responses.write" in msg or "401 Unauthorized" in msg:
            msg = (
                f"{msg} | Codex MCP needs an API key with 'api.responses.write' scope. "
                "Ensure OPENAI_API_KEY has the right permissions or run `codex login` with a key that does."
            )
        return json.dumps({"status": "error", "error": msg})


def build_codex_coder_agent(llm_config):
    """Factory to build an AutoGen agent that can call Codex via MCP."""
    try:
        import autogen
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "autogen is not installed. Install it with `pip install pyautogen`."
        ) from exc

    system_message = (
        "Codex_Coder. Use the tool `codex_mcp_run` for code edits and running commands. "
        "Provide a clear, actionable prompt for Codex that includes file paths and desired changes."
    )
    return autogen.AssistantAgent(
        name="Codex_Coder",
        system_message=system_message,
        llm_config=llm_config,
        function_map={"codex_mcp_run": codex_mcp_run},
    )
