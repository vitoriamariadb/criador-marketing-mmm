"""Configuracao centralizada de logging com loguru."""

import sys
from pathlib import Path

from loguru import logger

from src.config import LOG_FORMAT, LOG_RETENTION, LOG_ROTATION, LOGS_DIR


def setup_logger() -> None:
    """Configura logger com rotacao e formato padrao."""
    logger.remove()

    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level="INFO",
    )

    log_file: Path = LOGS_DIR / "mmm_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        format=LOG_FORMAT,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        level="DEBUG",
        encoding="utf-8",
    )


setup_logger()

# "Medir o que e mensuravel, tornar mensuravel o que nao e." - Galileu Galilei
