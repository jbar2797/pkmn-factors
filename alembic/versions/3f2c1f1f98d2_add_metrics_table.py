"""add metrics table

Revision ID: 3f2c1f1f98d2
Revises: fdcbe190a7e3
Create Date: 2025-08-23 21:05:00

"""

from typing import Sequence, Union

from alembic import op  # type: ignore[attr-defined]
import sqlalchemy as sa

from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3f2c1f1f98d2"
down_revision: Union[str, Sequence[str], None] = "fdcbe190a7e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "metrics",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column(
            "asof_ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("card_key", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "n_trades", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "n_signals", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("avg_return", sa.Float(), nullable=True),
        sa.Column("cum_return", sa.Float(), nullable=True),
        sa.Column("volatility", sa.Float(), nullable=True),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("params", postgresql.JSONB(), nullable=True),
    )

    # helpful indexes
    op.create_index(
        "ix_metrics_card_model_end",
        "metrics",
        ["card_key", "model_version", "period_end"],
        unique=False,
    )
    op.create_foreign_key(
        "metrics_card_key_fkey",
        "metrics",
        "cards",
        ["card_key"],
        ["card_key"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("metrics_card_key_fkey", "metrics", type_="foreignkey")
    op.drop_index("ix_metrics_card_model_end", table_name="metrics")
    op.drop_table("metrics")
