"""Add price_score and popularity_score to component_scores.

Revision ID: 002_add_price_popularity
Revises: 001_initial_schema
Create Date: 2026-02-24

Phase 2 normalization: all 7 ranking factors are now persisted,
versioned, and auditable in component_scores.

Previously, price_score and popularity_score were computed at
runtime in ranking.py. Now they are stored alongside the other
5 component scores, enabling:
  - Full version history via score_version + computed_at
  - Offline recalibration
  - Historical ranking audit
  - Deterministic snapshot of all 7 factors
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "002_add_price_popularity"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "component_scores",
        sa.Column("price_score", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "component_scores",
        sa.Column("popularity_score", sa.Float(), nullable=False, server_default="0.0"),
    )


def downgrade() -> None:
    op.drop_column("component_scores", "popularity_score")
    op.drop_column("component_scores", "price_score")
