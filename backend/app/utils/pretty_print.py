"""Terminal pretty-printer for ChatResponse dicts.

Usage (after receiving a response from POST /v1/chat):

    import json, httpx
    from app.utils.pretty_print import print_chat_response

    resp = httpx.post("http://localhost:8000/v1/chat",
                      json={"query": "best phone under 40k for photography"})
    print_chat_response(resp.json())

Or pipe curl output through Python:

    curl -s -X POST http://localhost:8000/v1/chat \\
      -H "Content-Type: application/json" \\
      -d '{"query": "compare s24 vs iphone 15"}' | python -c "
import sys, json
from app.utils.pretty_print import print_chat_response
print_chat_response(json.load(sys.stdin))"
"""

from __future__ import annotations

from typing import Any


def print_chat_response(response: dict[str, Any]) -> None:
    """Pretty-print a ChatResponse dict to stdout for terminal testing."""
    structured = response.get("structured_output") or {}
    intent_type = structured.get("intent_type", "unknown")
    session_id = response.get("session_id", "")
    trace_id = response.get("trace_id", "")
    source = response.get("source", "ai")
    metadata = structured.get("metadata") or {}

    W = 62
    print(f"\n{'=' * W}")
    sid_short = session_id[:20] + "..." if len(session_id) > 20 else session_id
    print(f"  SESSION : {sid_short}")
    print(f"  SOURCE  : {source}   INTENT: {intent_type}")
    if trace_id:
        print(f"  TRACE   : {trace_id}")
    if metadata:
        conf = metadata.get("confidence_score", "?")
        method = metadata.get("retrieval_method", "?")
        sources = metadata.get("sources") or []
        print(f"  CONFIDENCE: {conf}   RETRIEVAL: {method}")
        if sources:
            print(f"  SOURCES : {', '.join(str(s) for s in sources)}")
    print("=" * W)

    components = structured.get("components") or []
    if not components:
        print("\n  (no components — check intent or DB data)")
        print(f"\n{'=' * W}\n")
        return

    for component in components:
        ctype = component.get("component", "unknown")
        data = component.get("data") or {}

        if ctype == "recommendation_cards":
            _print_recommendations(data, W)

        elif ctype == "comparison_table":
            _print_comparison(data, W)

        elif ctype == "product_detail":
            _print_product_detail(data, W)

        elif ctype == "review_summary":
            _print_reviews(data, W)

        else:
            print(f"\n  [{ctype}]")
            print(f"  {data}")

    print(f"\n{'=' * W}\n")


def _print_recommendations(data: dict[str, Any], W: int) -> None:
    print(f"\n{'─' * W}")
    print("  RECOMMENDATIONS")
    print(f"{'─' * W}")
    if data.get("user_intent_summary"):
        print(f"  Intent: {data['user_intent_summary']}")

    for i, p in enumerate(data.get("ranked") or [], start=1):
        score = int(float(p.get("match_score") or 0) * 100)
        title = p.get("title") or p.get("canonical_title") or "Unknown"
        price = p.get("price")
        price_str = f"  ₹{price:,.0f}" if price else ""
        print(f"\n  {i}. {title}{price_str}")
        print(f"     Match: {score}/100")

        breakdown = p.get("match_score_breakdown") or {}
        if breakdown:
            scores_str = "  ".join(
                f"{k[:4]}:{v}" for k, v in breakdown.items() if v is not None
            )
            print(f"     Breakdown: {scores_str}")

        pros = p.get("pros") or []
        cons = p.get("cons") or []
        if pros:
            print(f"     Pros : {', '.join(pros[:3])}")
        if cons:
            print(f"     Cons : {', '.join(cons[:2])}")

        best = p.get("best_use_cases") or []
        not_for = p.get("not_suitable_for") or []
        if best:
            print(f"     Best for     : {', '.join(best)}")
        if not_for:
            print(f"     Not ideal for: {', '.join(not_for)}")

        reason = p.get("reason") or ""
        if reason:
            print(f"     Why  : {reason[:130]}")

    rationale = data.get("rationale") or ""
    if rationale:
        print(f"\n  Rationale: {rationale[:220]}")


def _print_comparison(data: dict[str, Any], W: int) -> None:
    print(f"\n{'─' * W}")
    print("  COMPARISON")
    print(f"{'─' * W}")

    products = data.get("products") or []
    titles = [p.get("title") or p.get("canonical_title") or p.get("slug", "?") for p in products]
    print(f"  {' vs '.join(titles)}")

    winners = data.get("per_category_winners") or {}
    if winners:
        print("\n  Category Winners:")
        for cat, winner in winners.items():
            print(f"    {cat:<14} → {winner}")

    diffs = (data.get("technical_diffs") or [])[:6]
    if diffs:
        print("\n  Key Technical Differences:")
        for d in diffs:
            attr = d.get("attribute", "?")
            expl = d.get("explanation") or ""
            print(f"    {attr:<20} {expl[:80]}")

    implications = (data.get("real_world_implications") or [])[:4]
    if implications:
        print("\n  In Practice:")
        for line in implications:
            print(f"    • {line}")

    tradeoffs = (data.get("tradeoffs") or [])[:3]
    if tradeoffs:
        print("\n  Tradeoffs:")
        for t in tradeoffs:
            print(f"    – {t}")

    verdict = data.get("overall_recommendation") or ""
    if verdict:
        print(f"\n  Verdict: {verdict[:220]}")


def _print_product_detail(data: dict[str, Any], W: int) -> None:
    print(f"\n{'─' * W}")
    print("  PRODUCT DETAIL")
    print(f"{'─' * W}")
    title = data.get("canonical_title") or data.get("product_slug") or "Unknown"
    print(f"  {title}")

    spec_fields = [
        ("chipset", "Chipset"),
        ("display_type", "Display"),
        ("display_hz", "Refresh rate"),
        ("battery_mah", "Battery"),
        ("charging_w", "Charging"),
        ("camera_mp_main", "Main camera"),
        ("sensor_size", "Sensor"),
        ("thermal_design", "Thermal"),
        ("battery_chemistry", "Battery chem"),
        ("ram_gb", "RAM"),
        ("storage_gb", "Storage"),
        ("weight_g", "Weight"),
    ]
    print()
    for field, label in spec_fields:
        val = data.get(field)
        if val is not None:
            print(f"  {label:<16} {val}")

    highlights = (data.get("feature_highlights") or [])[:6]
    if highlights:
        print("\n  Highlights:")
        for h in highlights:
            print(f"    • {h}")


def _print_reviews(data: dict[str, Any], W: int) -> None:
    summaries = data.get("summaries") or {}
    if not summaries:
        return
    print(f"\n{'─' * W}")
    print("  USER REVIEW CONSENSUS")
    print(f"{'─' * W}")
    for slug, s in summaries.items():
        if isinstance(s, dict):
            label = s.get("sentiment_label") or ""
            score = s.get("sentiment_score")
            score_str = f" ({score:.2f})" if score is not None else ""
            print(f"\n  {slug}{score_str}  {label}")
            praise = s.get("key_praise") or []
            complaints = s.get("key_complaints") or []
            if praise:
                print(f"    Praise    : {', '.join(praise[:3])}")
            if complaints:
                print(f"    Complaints: {', '.join(complaints[:2])}")
