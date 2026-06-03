import ast
from typing import Any


def _fallback_file_chunk(
    file_path: str,
    content: str,
    *,
    chunk_type: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload = (content or "").strip() or f"file:{file_path}"
    metadata: dict[str, Any] = {"file": str(file_path), "type": chunk_type}
    if error:
        metadata["error"] = error
    return {
        "chunk_id": f"file:{file_path}",
        "chunk": payload,
        "metadata": metadata,
    }


def extract_code_chunks(file_path: str, content: str) -> list[dict[str, Any]]:
    """
    Parse Python code and extract class/function chunks for semantic retrieval.
    """
    chunks: list[dict[str, Any]] = []
    
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        # If there's a syntax error, we just return the whole file as a single chunk
        return [
            _fallback_file_chunk(
                file_path,
                content,
                chunk_type="FileSyntaxError",
                error=str(exc),
            )
        ]

    lines = content.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = max(0, getattr(node, "lineno", 1) - 1)
        end_lineno = getattr(node, "end_lineno", None) or getattr(node, "lineno", 1)
        end = min(len(lines), max(start + 1, end_lineno))
        chunk = "\n".join(lines[start:end]).strip()
        if not chunk:
            continue

        chunk_id = f"{file_path}::{getattr(node, 'name', 'anonymous')}"
        metadata = {
            "file": str(file_path),
            "name": getattr(node, "name", ""),
            "type": type(node).__name__,
            "lines": f"{start + 1}-{end}",
        }
        chunks.append({
            "chunk_id": chunk_id,
            "chunk": chunk,
            "metadata": metadata
        })

    if not chunks:
        # If no functions or classes found, return the file as a single chunk
        chunks.append(
            _fallback_file_chunk(file_path, content, chunk_type="File")
        )

    return chunks
