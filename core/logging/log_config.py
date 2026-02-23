import logging
import sys
from pathlib import Path


_LOGGER_INITIALIZED = False


def setup(
    name: str = "Jarvis",
    level: str = "INFO",
    log_file: str | None = None,
):
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True
    logger.debug("Logging subsystem initialized")

    return logger


def get_logger(name: str = "Jarvis"):
    return logging.getLogger(name)


def _parse_level(level: str):
    if isinstance(level, int):
        return level

    level = level.upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(level, logging.INFO)
