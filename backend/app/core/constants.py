"""Evalio Core Constants — single source of truth for versioned constants.

Governance Rules:
    - EMBEDDING_VERSION: bump ONLY when changing embedding model, dimensions,
      or text serialization format. Bumping triggers full re-embed of all
      products (cost event). Requires team review.
    - Never auto-bump. Always increment by 1. Document reason in git commit.
    - If embedding dimension changes, a DB migration is also required:
        ALTER TABLE embeddings ALTER COLUMN embedding_vector TYPE vector(<new_dim>)

Operational Telemetry Notes:
    - Embedding cost metrics (total_generated, skip_rate etc.) are stored in Redis.
    - Redis restart = metrics reset. These are operational-only counters.
    - For long-term cost projection, pipe metrics to an external store (Phase 7+).
"""

from __future__ import annotations

# ── Embedding Version ─────────────────────────────────────────────────────────
# Increment only when: model changes, dimension changes, or text format changes.
# Current model: text-embedding-3-small (OpenAI) or text-embedding-004 (Gemini)
# Current dimension: 1536 (OpenAI) | 768 (Gemini)
#
# Version history:
#   v1 — Phase 6 initial: specs + review multi-embedding strategy
EMBEDDING_VERSION: int = 1

# ── Vector Dimension Lock ─────────────────────────────────────────────────────
# Embedding provider dimension must match this at embed time.
# If you change model, update BOTH constants + run Alembic migration.
OPENAI_EMBEDDING_DIM: int = 1536
GEMINI_EMBEDDING_DIM: int = 768
