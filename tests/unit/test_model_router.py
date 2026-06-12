import pytest
from unittest.mock import MagicMock
import configparser

from core.llm.model_router import ModelRouter
import os

@pytest.fixture
def router(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr("core.llm.model_router.ModelRouter.refresh_available_models", lambda self, **kw: None)
    r = ModelRouter()
    r._available_ollama_models = {"llama3.2:1b", "mistral:7b", "deepseek-r1:8b"}
    return r

def test_static_routing(router):
    router._strategy = "static"
    # Chat goes to tier 2
    assert router.route("chat") == "mistral:7b"
    # Deep reasoning goes to tier 3
    assert router.route("deep_reasoning") == "deepseek-r1:8b"
    # Reflex goes to tier 1
    assert router.route("reflex") == "llama3.2:1b"

def test_adaptive_routing_basic(router):
    router._strategy = "adaptive"
    classification = {
        "complexity": 0.4,
        "needs_reasoning": False,
        "needs_tools": False,
        "estimated_tokens": 100,
    }
    decision = router.route_adaptive(classification)
    # With complexity 0.4, we need min_reasoning 0.25. 
    # llama3.2:1b has 0.20 reasoning, so it's not enough.
    # qwen2.5:1.5b has 0.20
    # gemma2:2b has 0.22
    # mistral:7b has 0.55
    # So it should pick a tier 2 model (mistral:7b)
    assert decision.model == "mistral:7b"

def test_adaptive_routing_needs_tools(router):
    router._strategy = "adaptive"
    classification = {
        "complexity": 0.6,
        "needs_reasoning": False,
        "needs_tools": True,
        "estimated_tokens": 200,
    }
    decision = router.route_adaptive(classification)
    assert decision.model == "mistral:7b"
    
def test_adaptive_routing_needs_reasoning(router):
    router._strategy = "adaptive"
    classification = {
        "complexity": 0.9,
        "needs_reasoning": True,
        "needs_tools": False,
        "estimated_tokens": 500,
    }
    decision = router.route_adaptive(classification)
    assert decision.model == "deepseek-r1:8b"

def test_escalate(router):
    next_model = router.escalate("mistral:7b")
    # Should escalate from tier 2 to tier 3
    assert next_model == "deepseek-r1:8b"
    
def test_should_escalate(router):
    # Good response
    assert not router.should_escalate("mistral:7b", "chat", "Here is a detailed explanation of your question...")
    
    # Empty response
    assert router.should_escalate("mistral:7b", "chat", "")
    assert router.should_escalate("mistral:7b", "chat", "   ")
    
    # Short response for tier 2
    assert router.should_escalate("mistral:7b", "chat", "yes")
    
    # Refusal
    assert router.should_escalate("mistral:7b", "chat", "I cannot do that as an AI.")

def test_telemetry_integration(router):
    router._strategy = "adaptive"
    telemetry = MagicMock()
    # Mock reliable mistral, unreliable llama3
    def get_rel(m, task):
        if m == "llama3.2:1b":
            return 0.1
        if m == "mistral:7b":
            return 0.9
        return 0.5
    telemetry.get_reliability = MagicMock(side_effect=get_rel)
    router.set_telemetry(telemetry)
    
    classification = {
        "complexity": 0.1,  # normally would pick tier 1
        "needs_reasoning": False,
        "needs_tools": False,
        "estimated_tokens": 50,
    }
    decision = router.route_adaptive(classification)
    # Since qwen2.5:0.5b and llama3 are unreliable/not in mock, it will fallback or pick the highest reliable.
    # Actually, we should just ensure mistral:7b is picked because it's the only reliable one.
    assert decision.model == "mistral:7b"

def test_vision_routing(router):
    router._available_ollama_models.add("llava:latest")
    assert router.route("vision") == "llava:latest"

def test_config_override():
    config = configparser.ConfigParser()
    config.add_section("routing")
    config.set("routing", "strategy", "static")
    config.set("routing", "confidence_threshold", "0.8")
    
    router = ModelRouter(config=config)
    assert router.strategy == "static"
    assert router._confidence_threshold == 0.8
