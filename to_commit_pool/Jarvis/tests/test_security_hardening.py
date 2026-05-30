import os
import pytest
import configparser
from pathlib import Path

from core.security.auth import AuthManager
from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel
from core.tools.builtin_tools import _assert_safe_path, _SANDBOX_ROOT


def test_auth_manager_username_validation():
    db_path = Path("auth_test.db")
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass
    try:
        auth = AuthManager(db_path=db_path)

        # Valid user
        auth.create_user("valid_user", "securepassword123", is_admin=True)
        assert auth.authenticate("valid_user", "securepassword123") is not None

        # Username with pipe character
        with pytest.raises(ValueError, match="cannot contain the pipe character"):
            auth.create_user("invalid|user", "securepassword123")

        # Username with spaces
        with pytest.raises(ValueError, match="cannot contain spaces"):
            auth.create_user("invalid user", "securepassword123")
    finally:
        import gc
        gc.collect()
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass


def test_risk_evaluator_merges_config_instead_of_overwriting():
    # Build a config that only specifies high_risk_actions
    config = configparser.ConfigParser()
    config.add_section("risk")
    config.set("risk", "high_risk_actions", "custom_high_action")

    # Initialize RiskEvaluator with this config
    evaluator = RiskEvaluator(config=config)

    # Verify that default critical action (e.g. "delete_file") is still blocked
    res = evaluator.evaluate(["delete_file"])
    assert res.is_blocked
    assert res.level == RiskLevel.CRITICAL

    # Verify that the custom action is loaded and classified as HIGH
    res_custom = evaluator.evaluate(["custom_high_action"])
    assert res_custom.level == RiskLevel.HIGH


def test_windows_sandbox_case_insensitivity(monkeypatch):
    # Mock os.name to be "nt" to trigger case-insensitive check
    monkeypatch.setattr(os, "name", "nt")

    sandbox_path = str(_SANDBOX_ROOT)
    # Generate case variations
    cased_path = sandbox_path.lower() if sandbox_path.isupper() else sandbox_path.upper()

    try:
        res = _assert_safe_path(cased_path)
        assert res.resolve() == _SANDBOX_ROOT.resolve()
    except PermissionError:
        pytest.fail("Case variation of sandbox root was incorrectly blocked as escaping sandbox under NT")
    except ValueError:
        # ValueError represents target not in allowed legacy dirs list, which is expected
        pass
