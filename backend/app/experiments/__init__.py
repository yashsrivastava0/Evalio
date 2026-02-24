"""Evalio experiments — benchmarking, profiling, and optimization harnesses.

Available experiments:
    - embedding_experiment: Embedding model evaluation (NDCG, MRR, recall)
    - retrieval_benchmark_mock: Algorithm scaling test (synthetic data)
    - retrieval_benchmark_live: Live DB profiling (Docker PostgreSQL)
    - sql_explain: EXPLAIN ANALYZE for SQL filter + vector JOIN
    - candidate_cap_experiment: Vector/SQL cap optimization
    - ranking_benchmark: Ranking latency + determinism validation
    - load_test_simulation: Sequential load test (100/500/1000 queries)
"""
