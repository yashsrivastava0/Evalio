"""Phase 6 — Add embedding versioning columns to embeddings table.

Revision ID: 003
Revises: 002
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002_add_price_popularity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add versioning + hash columns to embeddings table
    op.add_column(
        "embeddings",
        sa.Column("embedding_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "embeddings",
        sa.Column("specs_hash", sa.Text(), nullable=True),
    )
    op.add_column(
        "embeddings",
        sa.Column("reviews_hash", sa.Text(), nullable=True),
    )

    # Composite index (product_id, embedding_type) for efficient per-type lookups
    op.create_index(
        "ix_embeddings_product_id_type",
        "embeddings",
        ["product_id", "embedding_type"],
        unique=False,
    )

    # Unique constraint to enforce one embedding per (product_id, embedding_type)
    # Required for ON CONFLICT upserts in the embedding pipeline
    op.create_unique_constraint(
        "uq_embeddings_product_type",
        "embeddings",
        ["product_id", "embedding_type"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_embeddings_product_type", "embeddings", type_="unique")
    op.drop_index("ix_embeddings_product_id_type", table_name="embeddings")
    op.drop_column("embeddings", "reviews_hash")
    op.drop_column("embeddings", "specs_hash")
    op.drop_column("embeddings", "embedding_version")
