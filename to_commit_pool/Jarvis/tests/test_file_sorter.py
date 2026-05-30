import shutil
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.tools.builtin_tools import (
    sort_files,
    find_files,
    copy_file,
    move_file,
    create_directory
)

@pytest.fixture
def test_workspace():
    # Setup test workspace directory inside ALLOWED_DIRECTORIES
    ws_path = Path("workspace/test_temp")
    if ws_path.exists():
        shutil.rmtree(ws_path)
    ws_path.mkdir(parents=True, exist_ok=True)
    yield ws_path
    if ws_path.exists():
        shutil.rmtree(ws_path)


@pytest.mark.anyio
async def test_create_directory(test_workspace):
    dir_path = test_workspace / "new_folder"
    res = await create_directory(str(dir_path))
    assert "Successfully created directory" in res
    assert dir_path.exists()
    assert dir_path.is_dir()


@pytest.mark.anyio
async def test_copy_and_move_file(test_workspace):
    src = test_workspace / "source.txt"
    src.write_text("Hello World", encoding="utf-8")
    
    dst_copy = test_workspace / "destination_copy.txt"
    res_copy = await copy_file(str(src), str(dst_copy))
    assert "Successfully copied" in res_copy
    assert dst_copy.exists()
    assert dst_copy.read_text(encoding="utf-8") == "Hello World"
    
    dst_move = test_workspace / "destination_move.txt"
    res_move = await move_file(str(src), str(dst_move))
    assert "Successfully moved" in res_move
    assert dst_move.exists()
    assert not src.exists()
    assert dst_move.read_text(encoding="utf-8") == "Hello World"


@pytest.mark.anyio
async def test_find_files(test_workspace):
    # Create subdirs and some files
    sub = test_workspace / "subdir"
    sub.mkdir()
    (test_workspace / "file1.txt").write_text("1")
    (sub / "file2.py").write_text("2")
    (sub / "other.txt").write_text("3")
    
    res = await find_files("*.txt", str(test_workspace))
    assert "[FILE] file1.txt" in res
    assert "[FILE] subdir" in res and "other.txt" in res
    assert "file2.py" not in res


@pytest.mark.anyio
async def test_sort_files_fallback(test_workspace):
    # Set _LLM_CLIENT to None to trigger fallback classification
    with patch("core.tools.builtin_tools._LLM_CLIENT", None):
        code_file = test_workspace / "script.py"
        code_file.write_text("print('hello')", encoding="utf-8")
        
        doc_file = test_workspace / "readme.md"
        doc_file.write_text("# Readme", encoding="utf-8")
        
        res = await sort_files(str(test_workspace), str(test_workspace))
        assert "Successfully sorted 2/2 files" in res
        assert (test_workspace / "code" / "script.py").exists()
        assert (test_workspace / "documentation" / "readme.md").exists()


@pytest.mark.anyio
async def test_sort_files_llm(test_workspace):
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=["logs", "finance"])
    
    with patch("core.tools.builtin_tools._LLM_CLIENT", mock_llm):
        file1 = test_workspace / "invoice.txt"
        file1.write_text("Invoice data 123", encoding="utf-8")
        
        file2 = test_workspace / "app.log"
        file2.write_text("ERROR: failed", encoding="utf-8")
        
        res = await sort_files(str(test_workspace), str(test_workspace))
        assert "Successfully sorted 2/2 files" in res
        assert (test_workspace / "finance" / "invoice.txt").exists()
        assert (test_workspace / "logs" / "app.log").exists()
