"""Legacy logging shim for older import paths."""

from core.logging import logger as _logger_module

AuditLog = _logger_module.AuditLog
setup = _logger_module.setup
get = _logger_module.get
get_logger = _logger_module.get_logger
audit = _logger_module.audit
verify_audit = _logger_module.verify_audit


def __getattr__(name: str):
    return getattr(_logger_module, name)


__all__ = [
    "AuditLog",
    "setup",
    "get",
    "get_logger",
    "audit",
    "verify_audit",
]
