from __future__ import annotations

import logging as py_logging

PREFIX = "[HareSkip]"

logger = py_logging.getLogger("HareSkip")
logger.setLevel(py_logging.INFO)


def _format(message: str) -> str:
    return f"{PREFIX} {message}"


def info(message: str) -> None:
    logger.info(_format(message))


def warning(message: str) -> None:
    logger.warning(_format(message))


def error(message: str) -> None:
    logger.error(_format(message))


def exception(message: str) -> None:
    logger.exception(_format(message))
