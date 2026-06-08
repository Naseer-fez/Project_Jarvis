import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ALLOWED_DIRECTORIES = [
    (_PROJECT_ROOT / "workspace").resolve(),
    (_PROJECT_ROOT / "outputs").resolve(),
]

# Project root sandbox - all resolved paths must stay inside here.
_SANDBOX_ROOT = _PROJECT_ROOT

def _assert_safe_path(path_str: str, write_op: bool = False) -> Path:
    """Raise PermissionError / ValueError if path is outside the sandbox."""
    # Block path traversal sequences before resolution
    if ".." in str(Path(path_str)):
        raise PermissionError(f"Path traversal blocked: {path_str}")

    resolved = Path(path_str).resolve()
    sandbox = _SANDBOX_ROOT

    resolved_str = str(resolved)
    sandbox_str = str(sandbox)
    if os.name == "nt":
        resolved_str = resolved_str.lower()
        sandbox_str = sandbox_str.lower()

    # Must be inside project sandbox
    if not (resolved_str == sandbox_str or resolved_str.startswith(sandbox_str + os.sep) or resolved_str.startswith(sandbox_str + "/")):
        raise PermissionError(f"Path outside sandbox: {resolved}")

    # Symlink must not escape sandbox
    if resolved.is_symlink():
        link_target = resolved.resolve()
        link_target_str = str(link_target)
        if os.name == "nt":
            link_target_str = link_target_str.lower()
        if not (link_target_str == sandbox_str or link_target_str.startswith(sandbox_str + os.sep) or link_target_str.startswith(sandbox_str + "/")):
            raise PermissionError(f"Symlink escapes sandbox: {link_target}")

    # Also check legacy ALLOWED_DIRECTORIES for backward compatibility
    target = resolved
    if write_op:
        allowed_dirs = ALLOWED_DIRECTORIES
    else:
        allowed_dirs = ALLOWED_DIRECTORIES + [
            (_PROJECT_ROOT / "config").resolve(),
            (_PROJECT_ROOT / "data").resolve(),
            (_PROJECT_ROOT / "logs").resolve(),
            (_PROJECT_ROOT / "core").resolve(),
        ]
    
    target_str = str(target)
    if os.name == "nt":
        target_str = target_str.lower()

    for allowed in allowed_dirs:
        allowed_str = str(allowed)
        if os.name == "nt":
            allowed_str = allowed_str.lower()
        if target_str == allowed_str or target_str.startswith(allowed_str + os.sep) or target_str.startswith(allowed_str + "/"):
            return target
        try:
            target.relative_to(allowed)
            return target
        except ValueError:
            continue
    raise ValueError(f"Path '{path_str}' is outside the sandbox. Allowed: {[str(d) for d in allowed_dirs]}")
