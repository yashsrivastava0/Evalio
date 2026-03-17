"""CLI Renderer — Terminal output formatting for Evalio CLI.

All print logic is extracted here. cli.py controls flow; this module only formats.
No state mutation. No graph invocation. Pure formatting.
"""

from __future__ import annotations

from typing import Any


# ── ANSI Colors ──────────────────────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"
_LINE = "─" * 50


def render_response(result: dict[str, Any], *, celery_available: bool = True) -> None:
    """Route to the correct renderer based on response source."""
    response = result.get("response", {})
    source = response.get("source", "unknown")

    if source == "fast_path":
        catalog_miss = response.get("catalog_miss_slugs")
        if catalog_miss:
            names = ", ".join(s.replace("-", " ").title() for s in catalog_miss)
            print(f"\n{_YELLOW}⚠  {names} not found in catalog — showing similar alternatives{_RESET}")
        render_ranked_results(response.get("ranked", []))
        render_explanation(response.get("explanation", {}))
    elif source == "feature_path":
        render_feature_specs(response.get("product", {}))
    elif source == "cache":
        render_cache_hit(response)
    elif source == "fetch_trigger":
        render_fetch_trigger(response, celery_available=celery_available)
    else:
        print(f"\n{_DIM}Evalio > {response}{_RESET}")


def render_ranked_results(ranked: list[dict[str, Any]]) -> None:
    """Print numbered ranking table with title, score, price."""
    if not ranked:
        print(f"\n{_DIM}No products ranked.{_RESET}")
        return

    print(f"\n{_BOLD}{_CYAN}RANKED RESULTS{_RESET}")
    print(_LINE)
    for i, p in enumerate(ranked, 1):
        title = p.get("canonical_title") or p.get("model_name") or p.get("product_slug", "Unknown")
        score = p.get("final_score", 0.0)
        price = p.get("best_price")
        price_str = f"₹{price:,.0f}" if price else "N/A"

        # Color rank position
        if i == 1:
            rank_color = _GREEN
        elif i <= 3:
            rank_color = _CYAN
        else:
            rank_color = _DIM

        print(f"  {rank_color}{i}. {title}{_RESET}")
        print(f"     Score: {_BOLD}{score:.3f}{_RESET}  |  Price: {price_str}")
    print(_LINE)


def render_feature_specs(product: dict[str, Any]) -> None:
    """Print structured specs block for a feature_lookup response."""
    if not product:
        print(f"\n{_DIM}No product data available.{_RESET}")
        return

    title = product.get("canonical_title") or product.get("model_name") or "Unknown"
    specs = product.get("specs_jsonb") or {}
    scores = product.get("scores") or {}

    print(f"\n{_BOLD}{_CYAN}PRODUCT SPECS{_RESET}: {title}")
    print(_LINE)

    if specs:
        for key, val in sorted(specs.items()):
            if val is not None:
                label = key.replace("_", " ").title()
                print(f"  {label:<26} {val}")
    else:
        print(f"  {_DIM}(No detailed specs available){_RESET}")

    if scores:
        print(f"\n  {_BOLD}Component Scores{_RESET}")
        for k, v in sorted(scores.items()):
            if isinstance(v, (int, float)):
                label = k.replace("_score", "").title()
                filled = int(v * 10)
                bar = _GREEN + "█" * filled + _DIM + "░" * (10 - filled) + _RESET
                print(f"  {label:<16} {bar} {_BOLD}{v:.2f}{_RESET}")

    print(_LINE)


def render_explanation(explanation: dict[str, Any]) -> None:
    """Print explanation summary — final verdict + source."""
    if not explanation:
        return

    source = explanation.get("source", "template")
    verdict = explanation.get("final_verdict", "")
    comparison = explanation.get("comparison_insight", "")

    print(f"\n{_BOLD}{_MAGENTA}EXPLANATION{_RESET} {_DIM}(source: {source}){_RESET}")
    print(_LINE)
    if verdict:
        print(f"  {verdict}")
    if comparison:
        print(f"\n  {_DIM}Comparison:{_RESET} {comparison}")
    print(_LINE)


def render_cache_hit(response: dict[str, Any]) -> None:
    """Print cached response."""
    print(f"\n{_BOLD}{_GREEN}CACHE HIT{_RESET}")
    print(_LINE)
    data = response.get("data", "")
    print(f"  {data}")
    print(_LINE)


def render_fetch_trigger(
    response: dict[str, Any],
    *,
    celery_available: bool = True,
) -> None:
    """Print fetch trigger status — matches response_node() contract.

    Issue #8: reads source, status, message, fetch_in_flight from response.
    Issue #2/Adjustment #1: warns if Celery is not running.
    """
    if not celery_available:
        print(f"\n{_YELLOW}{_BOLD}⚠️  Fetch trigger requested but Celery is not running.{_RESET}")
        print(f"  {_DIM}Start Celery worker to enable background data fetching.{_RESET}")
        return

    message = response.get("message", "Product data is being fetched. Please retry shortly.")
    in_flight = response.get("fetch_in_flight", False)

    print(f"\n{_BOLD}{_YELLOW}FETCH TRIGGERED{_RESET}")
    print(_LINE)
    print(f"  {message}")
    if in_flight:
        print(f"  {_DIM}(Scrape already in progress — will not re-enqueue){_RESET}")
    print(_LINE)


def render_budget_banner(result: dict[str, Any]) -> None:
    """Print trace ID, budget status, and request cost after each query.

    Reads from response.telemetry (Phase 9 GraphState fields).
    """
    response = result.get("response", {})
    trace_id = response.get("trace_id", result.get("trace_id", ""))
    telemetry = response.get("telemetry", result.get("telemetry", {}))

    # Compute request-level totals from telemetry spans
    total_tokens = 0
    total_cost = 0.0
    budget_exceeded = False
    for node_data in telemetry.values():
        if isinstance(node_data, dict):
            total_tokens += node_data.get("tokens_in", 0) + node_data.get("tokens_out", 0)
            total_cost += node_data.get("cost_usd", 0.0)
            if node_data.get("budget_exceeded"):
                budget_exceeded = True

    # Budget status color
    if budget_exceeded:
        budget_str = f"{_RED}{_BOLD}⚠️  BUDGET EXCEEDED{_RESET}"
    else:
        budget_str = f"{_GREEN}Within budget{_RESET}"

    print(f"\n{_DIM}{'═' * 50}{_RESET}")
    print(f"  Trace ID:  {_DIM}{trace_id[:16]}...{_RESET}" if len(trace_id) > 16 else f"  Trace ID:  {_DIM}{trace_id}{_RESET}")
    print(f"  Budget:    {budget_str}")
    print(f"  Cost:      {_BOLD}${total_cost:.6f}{_RESET}")
    print(f"  Tokens:    {total_tokens}")
    print(f"{_DIM}{'═' * 50}{_RESET}")


def render_telemetry_summary(telemetry: dict[str, Any]) -> None:
    """Print per-node latency + cost table from GraphState telemetry."""
    if not telemetry:
        return

    print(f"\n{_BOLD}NODE LATENCIES{_RESET}")
    print(f"  {'Node':<22} {'Duration':>10} {'Cost':>12} {'Tokens':>8}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 12} {'─' * 8}")

    for node_name, data in telemetry.items():
        if not isinstance(data, dict):
            continue
        dur = data.get("duration_ms", 0.0)
        cost = data.get("cost_usd", 0.0)
        tokens = data.get("tokens_in", 0) + data.get("tokens_out", 0)
        exceeded = " ⚠️" if data.get("budget_exceeded") else ""

        print(f"  {node_name:<22} {dur:>8.1f}ms ${cost:>10.6f} {tokens:>7}{exceeded}")


def render_cost_summary(cost_data: dict[str, float]) -> None:
    """Print session-level cost summary from Redis."""
    total_cost = cost_data.get("total_cost_usd", 0.0)
    total_tokens = cost_data.get("total_tokens", 0.0)

    print(f"\n{_BOLD}{_CYAN}SESSION COST SUMMARY{_RESET}")
    print(_LINE)
    print(f"  Total cost:   {_BOLD}${total_cost:.6f}{_RESET}")
    print(f"  Total tokens: {_BOLD}{int(total_tokens)}{_RESET}")
    print(_LINE)


def render_metrics(spans: list[dict[str, Any]], limit: int = 20) -> None:
    """Print last N telemetry spans from the per-process ring buffer.

    Issue #5: includes scope disclaimer — graph process only.
    """
    print(f"\n{_BOLD}{_CYAN}TELEMETRY SPANS{_RESET} {_DIM}(last {limit}, graph process only){_RESET}")
    print(f"  {_DIM}Note: Celery background task spans are logged separately.{_RESET}")
    print(f"  {'Node':<22} {'Duration':>10} {'Cost':>12} {'Tokens':>8} {'Trace':>12}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 12} {'─' * 8} {'─' * 12}")

    recent = spans[-limit:] if len(spans) > limit else spans
    for s in recent:
        node = s.get("node_name", "?")
        dur = s.get("duration_ms", 0.0)
        cost = s.get("cost_usd", 0.0)
        tokens = s.get("tokens_in", 0) + s.get("tokens_out", 0)
        trace = s.get("trace_id", "")[:12]

        print(f"  {node:<22} {dur:>8.1f}ms ${cost:>10.6f} {tokens:>7} {trace:>12}")

    print(f"\n  {_DIM}Buffer size: {len(spans)} / 100{_RESET}")


def render_startup_banner(session_id: str, celery_available: bool) -> None:
    """Print startup banner with session info and status."""
    print(f"\n{'═' * 50}")
    print(f" ⭐ {_BOLD}EVALIO AI COMMERCE ENGINE — CLI{_RESET} ⭐")
    print(f"{'═' * 50}")
    print(f"  Session:  {_CYAN}{session_id}{_RESET}")
    print(f"\n  Commands: {_DIM}:exit :session_reset :trace :help{_RESET}")
    print(f"{'═' * 50}\n")
