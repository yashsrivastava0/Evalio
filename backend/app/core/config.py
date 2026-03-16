"""Evalio AI Commerce Engine — Core configuration.

All settings are loaded from environment variables with sensible defaults.
Includes freshness TTLs, retrieval limits, and provider configuration.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # ── Database ──
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/antigravity"
    database_sync_url: str = "postgresql://postgres:postgres@localhost:5432/antigravity"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM Provider (pluggable) ──
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # ── Scraper ──
    proxy_url: str = ""
    raw_html_bucket: str = ""

    # ── Two-Tier Fetch Engine ──
    # Tier-2 allowed domains (comma-separated).
    # Only URLs within these domains are scraped by the discovery agent.
    tier2_allowed_domains: str = (
        "gsmarena.com,notebookcheck.net,amazon.in,flipkart.com,"
        "apple.com,samsung.com,oneplus.com,mi.com,xiaomi.com,"
        "realme.com,motorola.com,nokia.com,oppo.com,vivo.com,"
        "asus.com,dell.com,lenovo.com,hp.com,acer.com,msi.com,"
        "lg.com,sony.com,google.com,microsoft.com"
    )
    # Max candidate pages the Tier-2 agent will scrape per product
    tier2_max_discovery_pages: int = 3

    # ── Freshness TTLs ──
    price_stale_after_hours: int = 24
    review_stale_after_days: int = 7
    component_score_stale_after_days: int = 30

    # ── Retrieval pipeline limits (configurable) ──
    max_sql_candidates: int = 300
    max_vector_candidates: int = 50
    max_rank_candidates: int = 30

    # ── Celery ──
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Embedding ──
    embedding_dimension: int = 1536

    # ── Phase 9: Budget Ceilings ──
    # Read from config.py defaults; overridable via .env (review point #6).
    # Not hard-coded in modules — single source of truth here.
    max_cost_per_request_usd: float = 0.02
    max_tokens_per_request: int = 15_000
    max_cost_per_session_usd: float = 1.00

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()
