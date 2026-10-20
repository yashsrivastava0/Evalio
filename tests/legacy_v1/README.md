# legacy_v1

Tests written against the first-generation deterministic pipeline
(`ranking`, `decision_matrix`, `comparison_guard`, `explanation`, `router`, `cost`, `telemetry`),
which was replaced by the v2 multi-agent LangGraph pipeline.

They import modules that no longer exist, so `tests/conftest.py` excludes this folder from collection.
They are kept as a record of the original behaviour and as a reference when writing the v2 test suite.
