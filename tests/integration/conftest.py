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
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(url: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "app/db/migrations"))
    os.environ["ALEMBIC_URL"] = url
    return cfg


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    """A migrated, isolated SQLite database exposed as a session factory."""
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"

    command.upgrade(_alembic_config(url), "head")

    engine = create_engine(url, connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()
        os.environ.pop("ALEMBIC_URL", None)


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    """TestClient with `get_db` overridden to use the migrated test database."""
    from app.db.session import get_db
    from app.main import create_app

    def _override_get_db() -> Iterator[Session]:
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
