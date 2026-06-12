# API Analyst Report: tools\path_utils.py

## Dependencies
- `import os`
- `from pathlib import Path`

## Configuration Variables
- `_PROJECT_ROOT` = `Path(__file__).resolve().parent.parent.parent`
- `ALLOWED_DIRECTORIES` = `[(_PROJECT_ROOT / 'workspace').resolve(), (_PROJECT_ROOT / 'outputs').resolve()]`
- `_SANDBOX_ROOT` = `_PROJECT_ROOT`

## Functions & Endpoints

### `_assert_safe_path`
`def _assert_safe_path(path_str: str, write_op: bool=False) -> Path`
> Raise PermissionError / ValueError if path is outside the sandbox.
