import importlib as _importlib
import sys as _sys
_stdlib_logging = _importlib.import_module("logging")

logging = _stdlib_logging

log = logging.getLogger()

