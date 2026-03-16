# Hybrid Retrieval Agent

This agent performs pure DB + cache retrieval. No LLM is used.

## Responsibilities

1. Session bypass — reuse cached candidates for follow-up queries
2. Semantic cache check — return cached response if embedding matches
3. Entity resolution — resolve product names to DB slugs for compare queries
4. SQL filter — filter by category, budget, brand, RAM, storage
5. Vector rerank — HNSW cosine similarity on pgvector embeddings
6. Composite reranking — 0.4×semantic + 0.3×price + 0.2×brand + 0.1×recency
7. Deduplication — merge duplicate entries from multiple scrape sources

## Output

`retrieval_route` values:
- `cache_hit` — exact semantic cache match found
- `session_bypass` — reusing previous session candidates
- `entity_resolved` — COMPARE query with named products resolved to IDs
- `sql_vector` — full SQL filter + vector rerank pipeline
- `fetch_trigger` — DB has no matching products, trigger Celery scrape
