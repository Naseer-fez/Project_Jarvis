Write-Host "=== JARVIS LOGGER AUTO-FIX START ==="

# -------------------------------------------------
# 1. Rename logging.py (Python stdlib collision)
# -------------------------------------------------
if (Test-Path "core/logging/logging.py") {
    Rename-Item "core/logging/logging.py" "log_config.py" -Force
    Write-Host "Renamed logging.py -> log_config.py"
}

# -------------------------------------------------
# 2. Fix bad imports everywhere
# -------------------------------------------------
Get-ChildItem -Recurse -Filter *.py | ForEach-Object {
    (Get-Content $_.FullName) `
        -replace 'core\.logging\.logging', 'core.logging.log_config' `
        -replace 'from core\.logging import logging', 'from core.logging import log_config' |
    Set-Content $_.FullName
}

Write-Host "Fixed import references"

# -------------------------------------------------
# 3. Install HARDENED logger.py (defensive)
# -------------------------------------------------
@'
import logging
import sys
from pathlib import Path

_INITIALIZED = False


def setup(name="Jarvis", level="INFO", log_file=None):
    global _INITIALIZED

    # ---- HARD DEFENSE (prevents your crash) ----
    if not isinstance(name, str):
        name = "Jarvis"

    if not isinstance(level, (str, int)):
        level = "INFO"

    if log_file is not None and not isinstance(log_file, str):
        log_file = None
    # -------------------------------------------

    if _INITIALIZED:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _INITIALIZED = True
    logger.debug("Logging initialized safely")

    return logger


def get_logger(name="Jarvis"):
    if not isinstance(name, str):
        name = "Jarvis"
    return logging.getLogger(name)


def _parse_level(level):
    if isinstance(level, int):
        return level

    level = str(level).upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(level, logging.INFO)
'@ | Set-Content core/logging/logger.py

# -------------------------------------------------
# 4. Fix __init__.py
# -------------------------------------------------
@'
from .logger import setup, get_logger
'@ | Set-Content core/logging/__init__.py

Write-Host "Logger files rebuilt"

# -------------------------------------------------
# 5. Sanity check (must print function ref)
# -------------------------------------------------
python -c "from core.logging.logger import setup; print(setup)"

Write-Host "=== JARVIS LOGGER AUTO-FIX COMPLETE ==="