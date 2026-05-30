import pytest
from core.permission_matrix import PermissionMatrix, PermissionResult


class MockConfig:
    def __init__(self, settings):
        self.settings = settings

    def get(self, section, key, fallback=""):
        return self.settings.get(section, {}).get(key, fallback)


def test_permission_result_properties():
    """Verify PermissionResult helper properties."""
    p_empty = PermissionResult()
    assert p_empty.has_blocked is False
    assert p_empty.needs_confirmation is False

    p_blocked = PermissionResult(blocked_actions=["run_code"])
    assert p_blocked.has_blocked is True
    assert p_blocked.needs_confirmation is False

    p_confirm = PermissionResult(confirmation_actions=["delete_file"])
    assert p_confirm.has_blocked is False
    assert p_confirm.needs_confirmation is True


def test_permission_matrix_no_config():
    """Verify PermissionMatrix evaluation defaults to no blocked/confirmation with empty config."""
    pm = PermissionMatrix()
    res = pm.evaluate(["delete_file", "read_file"])
    assert res.blocked_actions == []
    assert res.confirmation_actions == []


def test_permission_matrix_with_config():
    """Verify PermissionMatrix correctly separates blocked and confirmation actions."""
    cfg = MockConfig({
        "risk": {
            "blocked_actions": "run_code, format_disk",
            "user_confirmed_actions": "delete_file, write_file"
        }
    })
    pm = PermissionMatrix(cfg)

    # All safe
    res = pm.evaluate(["read_file"])
    assert res.has_blocked is False
    assert res.needs_confirmation is False

    # Blocked action
    res = pm.evaluate(["format_disk"])
    assert res.blocked_actions == ["format_disk"]
    assert res.needs_confirmation is False

    # Confirmation action
    res = pm.evaluate(["delete_file"])
    assert res.blocked_actions == []
    assert res.confirmation_actions == ["delete_file"]

    # Mixed actions
    res = pm.evaluate(["read_file", "delete_file", "format_disk"])
    assert "format_disk" in res.blocked_actions
    assert "delete_file" in res.confirmation_actions
    assert "read_file" not in res.blocked_actions
    assert "read_file" not in res.confirmation_actions


def test_permission_matrix_case_insensitivity_and_spacing():
    """Verify that case and extra spaces are handled gracefully."""
    cfg = MockConfig({
        "risk": {
            "blocked_actions": "  Run_Code , Format_Disk  ",
            "user_confirmed_actions": "  Delete_File  "
        }
    })
    pm = PermissionMatrix(cfg)

    res = pm.evaluate(["  rUn_CoDe  ", "DELETE_FILE"])
    assert "run_code" in res.blocked_actions
    assert "delete_file" in res.confirmation_actions


def test_permission_matrix_blocked_fallback():
    """Verify blocked actions fall back to critical_actions and forbidden_actions."""
    cfg_critical = MockConfig({
        "risk": {
            "critical_actions": "reboot"
        }
    })
    pm_crit = PermissionMatrix(cfg_critical)
    assert pm_crit.evaluate(["reboot"]).blocked_actions == ["reboot"]

    cfg_forbidden = MockConfig({
        "risk": {
            "forbidden_actions": "uninstall"
        }
    })
    pm_forb = PermissionMatrix(cfg_forbidden)
    assert pm_forb.evaluate(["uninstall"]).blocked_actions == ["uninstall"]


def test_permission_matrix_confirmation_fallback():
    """Verify confirmation actions fall back to high_risk_actions."""
    cfg_high = MockConfig({
        "risk": {
            "high_risk_actions": "update_packages"
        }
    })
    pm = PermissionMatrix(cfg_high)
    assert pm.evaluate(["update_packages"]).confirmation_actions == ["update_packages"]
