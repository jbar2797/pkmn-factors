# mypy: ignore-errors
"""add features JSONB to signals

Revision ID: fdcbe190a7e3
Revises: cards_signals_mv_0001
Create Date: 2025-08-23 19:36:33.949386
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "fdcbe190a7e3"
down_revision: Union[str, Sequence[str], None] = "cards_signals_mv_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add signals.features as JSONB (nullable)."""
    op.add_column(
        "signals",
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema: drop signals.features."""
    op.drop_column("signals", "features")
