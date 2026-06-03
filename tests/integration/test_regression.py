from core.controller_v2 import JarvisControllerV2
import configparser

def test_missing_config_fallback():
    """Verify that JarvisControllerV2 falls back to defaults when config is missing or empty."""
    controller = JarvisControllerV2(config=None)
    
    # Assert sensible defaults
    assert controller.voice_enabled is False
    assert controller._gui_automation_enabled is False
    assert controller._app_launch_enabled is True
    
    assert controller.memory is not None
    assert controller.llm is not None
    
    # The default config should be created
    assert isinstance(controller.config, configparser.ConfigParser)

def test_controller_fallback_services():
    """Regression check: ensure fallback memory/llm doesn't crash initialization."""
    config = configparser.ConfigParser()
    controller = JarvisControllerV2(config=config)
    
    assert controller.memory is not None
    # Ensure memory can be accessed without exception
    assert hasattr(controller.memory, "mode")
