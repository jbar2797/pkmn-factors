# mypy: ignore-errors

"""Alembic environment configuration for async SQLAlchemy."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from typing import Any, Dict

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config, AsyncEngine

# --- Your app imports (must be importable) ---
# IMPORTANT: Base is declared in pkmn_factors.db.models
from pkmn_factors.db.models import Base  # SQLAlchemy declarative Base
from pkmn_factors.config import settings  # reads DATABASE_URL from env

# Alembic Config object, provides access to .ini values
config = context.config

# If you use logging via alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tell Alembic what metadata to scan for autogenerate
target_metadata = Base.metadata


def _alembic_config_with_url() -> Dict[str, Any]:
    """Prepare a config dict for async engine using env DATABASE_URL."""
    db_url = os.getenv("DATABASE_URL", settings.DATABASE_URL)
    cfg: Dict[str, Any] = config.get_section(config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = db_url
    return cfg


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no DB connection)."""
    url = os.getenv("DATABASE_URL", settings.DATABASE_URL)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Synchronous body called under both async/online and offline runs."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    cfg = _alembic_config_with_url()
    connectable: AsyncEngine = async_engine_from_config(cfg, prefix="sqlalchemy.")
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using async engine."""
    asyncio.run(run_async_migrations())


# Entry point used by alembic
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
