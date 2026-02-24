"""Initial schema — all 7 tables.

Revision ID: 001_initial_schema
Revises: None
Create Date: 2026-02-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enable pgvector ──
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Enums ──
    op.execute(
        "DO $$ BEGIN "
        "IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'product_category') THEN "
        "CREATE TYPE product_category AS ENUM ('smartphone', 'laptop'); "
        "END IF; END $$;"
    )
    op.execute(
        "DO $$ BEGIN "
        "IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'embedding_type') THEN "
        "CREATE TYPE embedding_type AS ENUM ('specs', 'review', 'summary'); "
        "END IF; END $$;"
    )

    # ── products ──
    op.create_table(
        "products",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("product_slug", sa.String(255), unique=True, nullable=False),
        sa.Column("brand", sa.String(100), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(50), nullable=False), # Placeholder for Enum
        sa.Column("release_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("specs_jsonb", JSONB, nullable=False, server_default="{}"),
        sa.Column("canonical_title", sa.String(500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    # Convert category to enum
    op.execute("ALTER TABLE products ALTER COLUMN category TYPE product_category USING category::product_category")
    
    op.create_index("ix_products_slug", "products", ["product_slug"])
    op.create_index("ix_products_brand", "products", ["brand"])
    op.create_index("ix_products_category", "products", ["category"])

    # ── variants ──
    op.create_table(
        "variants",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sku", sa.String(100), nullable=True),
        sa.Column("ram_gb", sa.Integer, nullable=True),
        sa.Column("storage_gb", sa.Integer, nullable=True),
        sa.Column("color", sa.String(50), nullable=True),
        sa.Column("price_inr", sa.Float, nullable=True),
        sa.Column("currency", sa.String(10), nullable=False, server_default="INR"),
        sa.Column("availability_bool", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("last_price_update", sa.DateTime(timezone=True), nullable=True),
        sa.Column("affiliate_url", sa.Text, nullable=True),
        sa.Column("vendor", sa.String(100), nullable=True),
    )
    op.create_index("ix_variants_product_id", "variants", ["product_id"])
    op.create_index("ix_variants_price", "variants", ["price_inr"])
    op.create_index("ix_variants_vendor", "variants", ["vendor"])
    op.create_index(
        "ix_variants_category_avail_price",
        "variants",
        ["availability_bool", "price_inr"],
    )

    # ── component_scores ──
    op.create_table(
        "component_scores",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id", ondelete="CASCADE"), nullable=False),
        sa.Column("performance_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("battery_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("camera_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("display_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("build_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("score_version", sa.Integer, nullable=False, server_default="1"),
    )
    op.create_index("ix_component_scores_product_id", "component_scores", ["product_id"])

    # ── embeddings (with pgvector) ──
    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id", ondelete="CASCADE"), nullable=False),
        sa.Column("variant_id", sa.Integer, sa.ForeignKey("variants.id", ondelete="SET NULL"), nullable=True),
        sa.Column("embedding_vector", sa.Text, nullable=False), 
        sa.Column("embedding_type", sa.String(50), nullable=False), # Placeholder for Enum
        sa.Column("embedding_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    # Convert embedding_type to enum
    op.execute("ALTER TABLE embeddings ALTER COLUMN embedding_type TYPE embedding_type USING embedding_type::embedding_type")
    
    # Alter column to use vector type
    op.execute("ALTER TABLE embeddings ALTER COLUMN embedding_vector TYPE vector(1536) USING embedding_vector::vector(1536)")
    op.create_index("ix_embeddings_product_id", "embeddings", ["product_id"])
    op.create_index("ix_embeddings_hash", "embeddings", ["embedding_hash"])

    # NOTE: HNSW index must be created via raw SQL because Alembic
    # does not natively support pgvector index operations.
    op.execute(
        """
        CREATE INDEX ix_embeddings_hnsw
        ON embeddings
        USING hnsw (embedding_vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # ── price_history ──
    op.create_table(
        "price_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("variant_id", sa.Integer, sa.ForeignKey("variants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("price_inr", sa.Float, nullable=False),
        sa.Column("vendor", sa.String(100), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_price_history_variant_id", "price_history", ["variant_id"])

    # ── review_summary ──
    op.create_table(
        "review_summary",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("summary_text", sa.Text, nullable=False, server_default=""),
        sa.Column("sentiment_score", sa.Float, nullable=True),
        sa.Column("source_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_synced", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_review_summary_product_id", "review_summary", ["product_id"])

    # ── search_query_cache ──
    op.create_table(
        "search_query_cache",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("query_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("response_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("ttl_ms", sa.Integer, nullable=False, server_default="300000"),
    )
    op.create_index("ix_search_query_cache_hash", "search_query_cache", ["query_hash"])


def downgrade() -> None:
    op.drop_table("search_query_cache")
    op.drop_table("review_summary")
    op.drop_table("price_history")
    op.drop_index("ix_embeddings_hnsw", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_table("component_scores")
    op.drop_table("variants")
    op.drop_table("products")

    op.execute("DROP TYPE IF EXISTS embedding_type")
    op.execute("DROP TYPE IF EXISTS product_category")
    op.execute("DROP EXTENSION IF EXISTS vector")
