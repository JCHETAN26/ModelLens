"""Alembic migration environment.

The database URL comes from `ALEMBIC_URL` if set, otherwise from app settings.
All ORM models are imported so `target_metadata` sees the full schema for
autogenerate and offline rendering.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings

# Import models for their side effect of registering tables on Base.metadata.
from app.db import models  # noqa: F401
from app.db.session import Base

config = context.config

target_metadata = Base.metadata


def _url() -> str:
    return os.environ.get("ALEMBIC_URL") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
