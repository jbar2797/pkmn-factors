"""DB defaults for trades + daily continuous aggregate

Revision ID: 7b9e2c8a5c2e
Revises: 3f2c1f1f98d2
Create Date: 2025-08-24 00:00:00
"""

from alembic import op  # type: ignore[attr-defined]
import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "7b9e2c8a5c2e"
down_revision = "3f2c1f1f98d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Sensible defaults
    op.execute(
        """
        ALTER TABLE trades
            ALTER COLUMN source   SET DEFAULT 'unknown',
            ALTER COLUMN currency SET DEFAULT 'USD';
        """
    )

    # 2) Canonicalize the daily CAGG (ensure `bucket` exists), drop/recreate
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_matviews WHERE schemaname = 'public' AND matviewname = 'trades_daily'
            ) THEN
                EXECUTE 'DROP MATERIALIZED VIEW IF EXISTS public.trades_daily CASCADE';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS public.trades_daily
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', "timestamp") AS bucket,
            card_key,
            AVG(price)::numeric(12,2) AS avg_price
        FROM public.trades
        GROUP BY 1, 2
        WITH NO DATA;
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_trades_daily_key_bucket
            ON public.trades_daily (card_key, bucket);
        """
    )

    # 3) Run the refresh OUTSIDE a transaction
    # Alembic helper: autocommit_block() opens a non-transactional block.
    with op.get_context().autocommit_block():
        op.execute(
            "CALL refresh_continuous_aggregate('public.trades_daily', NULL, NULL);"
        )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS public.trades_daily CASCADE;")
    op.execute(
        """
        ALTER TABLE trades
            ALTER COLUMN source   DROP DEFAULT,
            ALTER COLUMN currency DROP DEFAULT;
        """
    )
