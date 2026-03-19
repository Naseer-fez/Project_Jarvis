from __future__ import annotations

import time
from configparser import ConfigParser
from unittest.mock import MagicMock, patch

from core.introspection.health import HealthStatus, run_startup_health_check
from core.llm.model_router import ModelRouter


def test_model_router_resolves_installed_latest_tag():
    cfg = ConfigParser()
    cfg["models"] = {
        "chat_model": "mistral:7b",
        "summarize_model": "llama3.2:1b",
        "fallback_model": "mistral:7b",
    }
    router = ModelRouter(cfg)
    router._available_ollama_models = {"deepseek-r1:8b", "mistral:latest"}
    router._cache_time = time.time()

    assert router.is_available("mistral:7b") is True
    assert router.get_best_available("chat") == "mistral:latest"


def test_model_router_aliases_synthesis_and_fallback_tasks():
    cfg = ConfigParser()
    cfg["models"] = {
        "chat_model": "mistral:7b",
        "summarize_model": "llama3.2:1b",
        "quick_model": "gemma3:1b",
        "fallback_model": "gemini-2.5-flash",
    }
    router = ModelRouter(cfg)

    assert router.route("synthesis") == "llama3.2:1b"
    assert router.route("web_search_summary") == "gemma3:1b"
    assert router.route("tool_parameter_extraction") == "gemma3:1b"
    assert router.route("context_title_generation") == "gemma3:1b"
    assert router.route("final_response") == "mistral:7b"
    assert router.route("fallback") == "gemini-2.5-flash"


def test_model_router_prefers_quick_model_for_available_quick_tasks():
    cfg = ConfigParser()
    cfg["models"] = {
        "chat_model": "mistral:7b",
        "summarize_model": "llama3.2:1b",
        "quick_model": "gemma3:1b",
        "fallback_model": "gemini-2.5-flash",
    }
    router = ModelRouter(cfg)
    router._available_ollama_models = {"gemma3:latest", "mistral:latest"}
    router._cache_time = time.time()

    assert router.get_best_available("web_search_summary") == "gemma3:latest"
    assert router.get_best_available("tool_parameter_extraction") == "gemma3:latest"


def test_controller_v2_sets_llm_router():
    with patch("core.controller_v2.HybridMemory") as mock_mem, \
         patch("core.controller_v2.ModelRouter") as mock_router, \
         patch("core.controller_v2.UserProfileEngine"), \
         patch("core.controller_v2.GoalManager"), \
         patch("core.controller_v2.Scheduler"), \
         patch("core.controller_v2.NotificationManager"), \
         patch("core.controller_v2.BackgroundMonitor"), \
         patch("core.controller_v2.ProfileSynthesizer"):

        router = MagicMock()
        router.route.return_value = "mistral:7b"
        mock_router.return_value = router
        mock_mem.return_value.initialize.return_value = {"mode": "lite"}

        from core.controller_v2 import JarvisControllerV2

        ctrl = JarvisControllerV2()

    assert ctrl.llm.model_router is router


def test_startup_health_reports_existing_sqlite(tmp_path):
    db_path = tmp_path / "jarvis_memory.db"
    db_path.write_text("", encoding="utf-8")
    controller = MagicMock()
    controller.memory.db_path = str(db_path)

    with patch("urllib.request.urlopen", return_value=object()):
        report = run_startup_health_check(controller)

    memory_check = next(check for check in report.checks if check.name == "memory_sqlite")
    assert memory_check.status == HealthStatus.OK
    assert memory_check.message == "True"
