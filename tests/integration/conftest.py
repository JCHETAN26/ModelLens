"""Shared DB fixtures: a fresh, migrated SQLite database per test.

We apply real Alembic migrations (not metadata.create_all) so the migration
scripts themselves are exercised on every run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(url: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "app/db/migrations"))
    os.environ["ALEMBIC_URL"] = url
    return cfg


@pytest.fixture
def db_session(tmp_path: Path) -> Iterator[Session]:
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"

    command.upgrade(_alembic_config(url), "head")

    engine = create_engine(url, connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        os.environ.pop("ALEMBIC_URL", None)
