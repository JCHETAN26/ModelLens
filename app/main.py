"""ModelLens FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.api import routes_cases, routes_explanations, routes_health
from app.core.errors import ModelLensError, modellens_error_handler
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Application factory. Keeps wiring explicit and test-friendly."""
    configure_logging()

    app = FastAPI(
        title="ModelLens",
        version=__version__,
        description=(
            "Convert structured risk model outputs into faithful, "
            "member-facing explanations with automated evaluation."
        ),
    )

    app.add_exception_handler(ModelLensError, modellens_error_handler)

    app.include_router(routes_health.router)
    app.include_router(routes_cases.router)
    app.include_router(routes_explanations.router)

    return app


app = create_app()
