# mypy: ignore-errors
"""init trades table + timescale extension

Revision ID: 336346fb1797
Revises:
Create Date: 2025-08-23 17:54:54.233989
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "336346fb1797"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # --- base table ---
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("card_key", sa.String(length=128), nullable=False),
        sa.Column("grade", sa.String(length=16), nullable=True),
        sa.Column("listing_type", sa.String(length=16), nullable=True),
        sa.Column("price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("link", sa.Text(), nullable=True),
        # COMPOSITE PRIMARY KEY must include partition column `timestamp`
        sa.PrimaryKeyConstraint("id", "timestamp"),
        sa.UniqueConstraint(
            "timestamp", "price", "source", "link", name="uq_trade_dedup"
        ),
    )
    op.create_index(
        "ix_trades_card_time", "trades", ["card_key", "timestamp"], unique=False
    )
    op.create_index(op.f("ix_trades_timestamp"), "trades", ["timestamp"], unique=False)

    # --- TimescaleDB bits ---
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    # Turn the plain table into a hypertable partitioned by `timestamp`
    op.execute("SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE)")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_trades_timestamp"), table_name="trades")
    op.drop_index("ix_trades_card_time", table_name="trades")
    op.drop_table("trades")
