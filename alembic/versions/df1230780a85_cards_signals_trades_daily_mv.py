# mypy: ignore-errors
"""cards + signals + trades_daily MV

Revision ID: cards_signals_mv_0001
Revises: 336346fb1797
Create Date: 2025-08-23

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "cards_signals_mv_0001"
down_revision: Union[str, Sequence[str], None] = "336346fb1797"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- cards ---
    op.create_table(
        "cards",
        sa.Column("card_key", sa.String(128), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("set_code", sa.String(64), nullable=True),
        sa.Column("set_name", sa.String(128), nullable=True),
        sa.Column("rarity", sa.String(64), nullable=True),
        sa.Column("rarity_bucket", sa.String(32), nullable=True),
        sa.Column("character", sa.String(64), nullable=True),
        sa.Column("art_type", sa.String(32), nullable=True),  # e.g. 'alt', 'promo'
        sa.Column("language", sa.String(16), nullable=True, server_default="EN"),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
    )
    op.create_index("ix_cards_character", "cards", ["character"])
    op.create_index("ix_cards_rarity_bucket", "cards", ["rarity_bucket"])

    # --- signals ---
    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "asof_ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("card_key", sa.String(128), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="90"),
        sa.Column("action", sa.String(8), nullable=False),  # BUY/HOLD/SELL
        sa.Column("conviction", sa.Float(), nullable=False),  # 0-100
        sa.Column(
            "expected_return", sa.Float(), nullable=False
        ),  # fractional (e.g. 0.05)
        sa.Column("risk", sa.Float(), nullable=False),  # fractional stdev
        sa.Column("utility", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=True),  # serialized features
        sa.ForeignKeyConstraint(["card_key"], ["cards.card_key"], ondelete="CASCADE"),
    )
    op.create_index("ix_signals_card_key_ts", "signals", ["card_key", "asof_ts"])

    # --- trades_daily materialized view ---
    # NOTE: Alembic runs inside a transaction; avoid CONCURRENTLY.
    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS trades_daily AS
        SELECT
            card_key,
            date(timestamp) AS day,
            COUNT(*)                   AS trade_count,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS median_price,
            AVG(price)                 AS mean_price,
            MAX(timestamp)             AS last_ts
        FROM trades
        GROUP BY card_key, day;
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_trades_daily_ck_day ON trades_daily(card_key, day);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_trades_daily_ck_day ON trades_daily(card_key, day);"
    )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS trades_daily;")
    op.drop_index("ix_signals_card_key_ts", table_name="signals")
    op.drop_table("signals")
    op.drop_index("ix_cards_rarity_bucket", table_name="cards")
    op.drop_index("ix_cards_character", table_name="cards")
    op.drop_table("cards")
