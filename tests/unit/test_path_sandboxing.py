import pytest
from core.tools.builtin_tools import _assert_safe_path, _PROJECT_ROOT, fast_search, convert_file_format
from core.tools.system_automation import write_file, delete_file, list_directory, read_file, execute_shell, launch_application


def test_assert_safe_path_valid():
    # Workspace should always be valid for read and write
    workspace_path = _PROJECT_ROOT / "workspace" / "test_file.txt"
    assert _assert_safe_path(str(workspace_path), write_op=True) == workspace_path.resolve()
    assert _assert_safe_path(str(workspace_path), write_op=False) == workspace_path.resolve()

    # Config should be valid for read-only
    config_path = _PROJECT_ROOT / "config" / "jarvis.ini"
    assert _assert_safe_path(str(config_path), write_op=False) == config_path.resolve()


def test_assert_safe_path_invalid_escapes():
    # Traversal attempt
    with pytest.raises(PermissionError):
        _assert_safe_path(str(_PROJECT_ROOT / ".." / "escaped.txt"))

    # Prefix mismatch escape attempt (sibling path bypass)
    sibling_path = str(_PROJECT_ROOT) + "_extra"
    with pytest.raises(PermissionError):
        _assert_safe_path(sibling_path)

    # Prefix mismatch escape attempt for allowed dirs (e.g. workspace_extra)
    workspace_sibling = str(_PROJECT_ROOT / "workspace") + "_extra"
    with pytest.raises((PermissionError, ValueError)):
        _assert_safe_path(workspace_sibling)

    # Read-only folder during write operations
    config_path = _PROJECT_ROOT / "config" / "jarvis.ini"
    with pytest.raises((PermissionError, ValueError)):
        _assert_safe_path(str(config_path), write_op=True)


@pytest.mark.asyncio
async def test_system_automation_sandboxing():
    # Test list_directory sandboxing
    res = list_directory(str(_PROJECT_ROOT / ".." / "escaped"))
    assert not res.success
    assert any(term in res.error.lower() for term in ("sandbox", "permission", "blocked", "outside", "escape"))

    # Test read_file sandboxing
    res = read_file(str(_PROJECT_ROOT / ".." / "escaped" / "file.txt"))
    assert not res.success
    assert any(term in res.error.lower() for term in ("sandbox", "permission", "blocked", "outside", "escape"))

    # Test write_file sandboxing
    res = write_file(str(_PROJECT_ROOT / ".." / "escaped" / "file.txt"), "content")
    assert not res.success
    assert any(term in res.error.lower() for term in ("sandbox", "permission", "blocked", "outside", "escape"))

    # Test delete_file sandboxing
    res = delete_file(str(_PROJECT_ROOT / ".." / "escaped" / "file.txt"))
    assert not res.success
    assert any(term in res.error.lower() for term in ("sandbox", "permission", "blocked", "outside", "escape"))

    # Test execute_shell sandboxing of working_dir
    res = await execute_shell("echo hello", working_dir=str(_PROJECT_ROOT / ".." / "escaped"))
    assert not res.success
    assert "escapes sandbox" in res.error

    # Test launch_application sandboxing of target
    res = launch_application(target=str(_PROJECT_ROOT / ".." / "escaped" / "app.exe"))
    assert not res.success
    assert "escapes sandbox" in res.error

    # Test launch_application sandboxing of target using '..' directly
    res = launch_application(target="..")
    assert not res.success
    assert "escapes sandbox" in res.error

    # Test launch_application sandboxing of arguments
    res = launch_application(target="notepad.exe", args=[str(_PROJECT_ROOT / ".." / "escaped" / "file.txt")])
    assert not res.success
    assert "escapes sandbox" in res.error

    # Test launch_application sandboxing of arguments using '..' directly
    res = launch_application(target="notepad.exe", args=[".."])
    assert not res.success
    assert "escapes sandbox" in res.error



@pytest.mark.asyncio
async def test_builtin_tools_sandboxing():
    # Test fast_search sandboxing
    res = await fast_search(path=str(_PROJECT_ROOT / ".." / "escaped"), query="*.py")
    assert "outside the sandbox" in res

    # Test convert_file_format sandboxing
    res = await convert_file_format(
        source_path=str(_PROJECT_ROOT / ".." / "escaped" / "src.txt"),
        target_format="pdf"
    )
    assert "outside sandbox" in res
