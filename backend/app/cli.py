"""Evalio CLI — Multi-turn terminal engine for the AI commerce system.

Entry point: python -m app.cli

Architecture:
    - Bootstrap via run_local.create_app() (DB, Redis, LLM, Embedding, Graph)
    - Interactive loop with command dispatcher
    - Fresh GraphState per query (Issue #4: never carry accumulated cost)
    - Commands (:exit, :session_reset, :trace) isolated from graph (Issue #7)
    - Async boundary via asyncio.run() (Issue #1)
"""

from __future__ import annotations

import asyncio
import traceback
from uuid import uuid4

from app.run_local import create_app, shutdown
from app.cli_renderer import render_startup_banner
from app.utils.pretty_print import print_chat_response


# ── Command Handlers ─────────────────────────────────────────────────────────
# Issue #7: Commands operate outside graph, never touch GraphState.


async def handle_command(cmd: str, ctx: dict) -> bool:
    """Dispatch CLI commands. Returns True if should exit."""
    cmd = cmd.strip().lower()

    if cmd in (":exit", ":quit", ":q"):
        await shutdown(ctx)
        return True

    elif cmd == ":session_reset":
        await _cmd_session_reset(ctx)

    elif cmd == ":trace":
        _cmd_trace(ctx)

    elif cmd == ":help":
        _cmd_help()

    else:
        print(f"  Unknown command: {cmd}")
        print(f"  Available: :exit :session_reset :trace :help")

    return False


async def _cmd_session_reset(ctx: dict) -> None:
    """Reset session: delete old Redis key, generate new UUID."""
    old_id = ctx["session_id"]
    redis = ctx["redis"]

    try:
        await redis.delete(f"session:{old_id}")
    except Exception:
        pass  # Non-blocking

    new_id = str(uuid4())
    ctx["session_id"] = new_id

    print(f"\n  Session reset.")
    print(f"  Old: {old_id}")
    print(f"  New: {new_id}")


def _cmd_trace(ctx: dict) -> None:
    """Show last trace_id."""
    trace_id = ctx.get("last_trace_id", "(no queries yet)")
    print(f"\n  Last trace_id: {trace_id}")


def _cmd_help() -> None:
    """Print available commands."""
    print("\n  Commands:")
    print("    :exit           — Close connections and exit")
    print("    :session_reset  — Reset session (new ID)")
    print("    :trace          — Show last trace_id")
    print("    :help           — Show this help")


# ── Main Loop ────────────────────────────────────────────────────────────────


async def main():
    """Async CLI entry point."""
    print("\n  Initializing Evalio...")
    ctx = await create_app()

    render_startup_banner(ctx["session_id"], ctx["celery_available"])

    while True:
        try:
            query = input("You > ").strip()

            if not query:
                continue

            # Command dispatch
            if query.startswith(":"):
                should_exit = await handle_command(query, ctx)
                if should_exit:
                    break
                continue

            print("  Evalio thinking...")

            # Fresh GraphState per query
            input_state = {
                "query": query,
                "session_id": ctx["session_id"],
            }

            result = await ctx["graph"].ainvoke(input_state)

            # Extract the ChatResponse dict from graph output
            response = result.get("response") or {}

            # Store trace_id for :trace command
            ctx["last_trace_id"] = response.get("trace_id", result.get("trace_id", ""))

            # Pretty-print using the structured renderer
            print_chat_response(response)

        except KeyboardInterrupt:
            print("\n  Session interrupted.")
            await shutdown(ctx)
            break
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
