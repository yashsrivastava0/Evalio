"""Test configuration and shared fixtures."""

from __future__ import annotations

import pytest

# Tests written against the pre-v2 deterministic pipeline (ranking, decision matrix,
# comparison guard, explanation). Kept for reference; not collected.
collect_ignore = ["legacy_v1"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
