"""Structured-ish logging setup.

Keeps configuration centralized so every module logs consistently.
"""

from __future__ import annotations

import logging

from app.core.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure root logging once, based on settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    configure_logging()
    return logging.getLogger(name)
