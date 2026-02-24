"""SQLAlchemy ORM models for all 7 database tables.

Tables:
    - products: Core product catalog (smartphones, laptops)
    - variants: SKU-level variants with pricing
    - component_scores: Deterministic scoring per product (versioned)
    - embeddings: pgvector embeddings with HNSW index
    - price_history: Historical price tracking
    - review_summary: Aggregated review data
    - search_query_cache: Audit/cache table

Indexes:
    - Composite: (category, availability_bool, price_inr) on variants via products
    - HNSW: embedding_vector on embeddings table
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


# ── Enums ──


class ProductCategory(str, enum.Enum):
    """Supported product categories — MVP: smartphones + laptops only."""

    smartphone = "smartphone"
    laptop = "laptop"


class EmbeddingType(str, enum.Enum):
    """Type of embedding stored — drives multi-embedding strategy."""

    specs = "specs"
    review = "review"
    summary = "summary"


# ── Helper ──


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Tables ──


class Product(Base):
    """Core product catalog.

    One row per unique product (e.g., 'iPhone 15 Pro').
    Variants hold SKU-level differences (RAM, storage, color, price).
    """

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    brand: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[ProductCategory] = mapped_column(
        Enum(ProductCategory, name="product_category", create_constraint=True),
        nullable=False,
        index=True,
    )
    release_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    specs_jsonb: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    canonical_title: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    variants: Mapped[list["Variant"]] = relationship(
        "Variant", back_populates="product", cascade="all, delete-orphan"
    )
    component_scores: Mapped[list["ComponentScore"]] = relationship(
        "ComponentScore", back_populates="product", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="product", cascade="all, delete-orphan"
    )
    review_summaries: Mapped[list["ReviewSummary"]] = relationship(
        "ReviewSummary", back_populates="product", cascade="all, delete-orphan"
    )


class Variant(Base):
    """SKU-level variant of a product.

    Tracks RAM, storage, color, current price, vendor, and availability.
    """

    __tablename__ = "variants"
    __table_args__ = (
        Index(
            "ix_variants_category_avail_price",
            "availability_bool",
            "price_inr",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sku: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ram_gb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    storage_gb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    color: Mapped[str | None] = mapped_column(String(50), nullable=True)
    price_inr: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="INR")
    availability_bool: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_price_update: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    affiliate_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    vendor: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="variants")
    price_history: Mapped[list["PriceHistory"]] = relationship(
        "PriceHistory", back_populates="variant", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="variant", cascade="all, delete-orphan"
    )


class ComponentScore(Base):
    """Deterministic component scores for ranking.

    Versioned via score_version + computed_at for staleness checks.
    These scores drive the deterministic ranking formula — no LLM involvement.
    """

    __tablename__ = "component_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    performance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    battery_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    camera_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    display_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    build_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    popularity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    score_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="component_scores")


class Embedding(Base):
    """pgvector embeddings for semantic search.

    Multi-embedding strategy: specs, review, summary vectors stored separately.
    HNSW index for fast approximate nearest-neighbor search.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        Index(
            "ix_embeddings_hnsw",
            "embedding_vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding_vector": "vector_cosine_ops"},
        ),
        # Composite index for per-type lookups — prevents cross-type scans
        Index(
            "ix_embeddings_product_id_type",
            "product_id",
            "embedding_type",
        ),
        # Unique constraint enables ON CONFLICT upserts in embedding pipeline
        UniqueConstraint("product_id", "embedding_type", name="uq_embeddings_product_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    variant_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("variants.id", ondelete="SET NULL"), nullable=True
    )
    embedding_vector = mapped_column(Vector(1536), nullable=False)
    embedding_type: Mapped[EmbeddingType] = mapped_column(
        Enum(EmbeddingType, name="embedding_type", create_constraint=True),
        nullable=False,
    )
    embedding_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Phase 6 — versioning + hash dedupe
    embedding_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    specs_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviews_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="embeddings")
    variant: Mapped["Variant | None"] = relationship("Variant", back_populates="embeddings")


class PriceHistory(Base):
    """Historical price tracking for variants.

    Enables price trend analysis and freshness gating.
    """

    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("variants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    price_inr: Mapped[float] = mapped_column(Float, nullable=False)
    vendor: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    variant: Mapped["Variant"] = relationship("Variant", back_populates="price_history")


class ReviewSummary(Base):
    """Aggregated review summary per product.

    Stores LLM-generated summary, sentiment score, and source count.
    """

    __tablename__ = "review_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_synced: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="review_summaries")


class SearchQueryCache(Base):
    """Audit / cache table for search queries.

    Stores query_hash → response for exact-match query caching.
    """

    __tablename__ = "search_query_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    response_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ttl_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=300_000)
