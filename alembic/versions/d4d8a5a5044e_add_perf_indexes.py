"""add perf indexes

Revision ID: d4d8a5a5044e
Revises: 3f2c1f1f98d2
Create Date: 2025-08-24 16:24:25.237991

"""

from typing import Sequence, Union

from alembic import op  # type: ignore[attr-defined]


# revision identifiers, used by Alembic.
revision: str = "d4d8a5a5044e"
down_revision: Union[str, Sequence[str], None] = "3f2c1f1f98d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # trades: most queries filter by card_key and time
    op.create_index(
        "ix_trades_card_key_timestamp",
        "trades",
        ["card_key", "timestamp"],
        unique=False,
    )

    # signals: filter by card_key, model_version, and order by asof_ts
    op.create_index(
        "ix_signals_card_model_asof_ts",
        "signals",
        ["card_key", "model_version", "asof_ts"],
        unique=False,
    )

    # metrics: frequently ordered by asof_ts
    op.create_index(
        "ix_metrics_asof_ts",
        "metrics",
        ["asof_ts"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_metrics_asof_ts", table_name="metrics")
    op.drop_index("ix_signals_card_model_asof_ts", table_name="signals")
    op.drop_index("ix_trades_card_key_timestamp", table_name="trades")
