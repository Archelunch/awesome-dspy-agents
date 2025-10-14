from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog  # type: ignore

_CONFIGURED: bool = False


def configure_logging(pretty_console: Optional[bool] = None) -> None:
    """Configure structlog once for the entire application.

    - Pretty console by default when TTY or MAD_LOG_PRETTY is truthy.
    - JSON rendering otherwise (useful for production/log files).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if pretty_console is None:
        env_val = os.getenv("MAD_LOG_PRETTY", "1").lower()
        pretty_console = env_val not in ("0", "false", "no") and sys.stderr.isatty()

    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if pretty_console:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(
    name: str,
    file_name: str,
    *,
    max_bytes: int = 2_000_000,
    backup_count: int = 3,
    logs_dir: Optional[Path] = None,
):
    """Return a structlog logger and ensure a rotating file handler is attached.

    The file handler is attached to the stdlib logger with the same name so that
    structlog's output is written as line-delimited JSON (or pretty text) into
    the specified file.
    """
    configure_logging()

    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / file_name

    std_logger = logging.getLogger(name)
    # Avoid duplicating the same file handler if called multiple times
    has_file = False
    for h in std_logger.handlers:
        try:
            if isinstance(h, RotatingFileHandler) and getattr(
                h, "baseFilename", None
            ) == str(log_path):
                has_file = True
                break
        except Exception:
            continue

    if not has_file:
        handler = RotatingFileHandler(
            str(log_path), maxBytes=max_bytes, backupCount=backup_count
        )
        # structlog renders into the message field
        handler.setFormatter(logging.Formatter("%(message)s"))
        std_logger.setLevel(logging.INFO)
        std_logger.addHandler(handler)

    return structlog.get_logger(name)
