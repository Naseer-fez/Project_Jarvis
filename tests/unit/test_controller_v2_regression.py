import configparser
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.controller_v2 import JarvisControllerV2

@pytest.fixture
def mock_config():
    config = configparser.ConfigParser()
    config.add_section("execution")
    config.set("execution", "allow_gui_automation", "false")
    config.set("execution", "allow_app_launch", "true")
    config.add_section("memory")
    config.add_section("automation")
    config.add_section("voice")
    return config

@pytest.fixture
def mock_services():
    services = MagicMock()
    services.memory = AsyncMock()
    services.memory.initialize.return_value = {"mode": "local"}
    services.model_router = MagicMock()
    services.profile = MagicMock()
    services.llm = AsyncMock()
    services.synthesizer = MagicMock()
    services.synthesizer.should_run.return_value = False
    services.synthesizer.synthesize = AsyncMock()
    services.state_machine = MagicMock()
    services.task_planner = MagicMock()
    services.tool_router = MagicMock()
    services.risk_evaluator = MagicMock()
    services.autonomy_governor = MagicMock()
    services.agent_loop = MagicMock()
    services.goal_manager = MagicMock()
    services.goal_manager.active_goals.return_value = []
    services.scheduler = MagicMock()
    services.notifier = AsyncMock()
    services.monitor = AsyncMock()
    services.desktop_executor = MagicMock()
    services.desktop_observer = MagicMock()
    services.desktop_bridge = MagicMock()
    services.container = MagicMock()
    return services

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.goals_file = "dummy_goals.json"
    settings.goal_check_interval_seconds = 60
    return settings

@pytest.fixture
def controller(mock_config, mock_services, mock_settings):
    with patch("core.controller_v2.build_controller_services") as mock_build:
        mock_build.return_value = (mock_settings, mock_services)
        ctrl = JarvisControllerV2(
            config=mock_config,
            services=mock_services,
            settings=mock_settings,
        )
        # Mock internal facades to prevent actual initialization issues
        ctrl.goal_runner = AsyncMock()
        ctrl.intent_router = AsyncMock()
        ctrl.intent_router.route.return_value = None  # Force LLM dispatch by default
        ctrl.llm_orchestrator = AsyncMock()
        ctrl.llm_orchestrator.dispatch.return_value = "LLM Response"
        yield ctrl

@pytest.mark.asyncio
async def test_controller_initialization(mock_config, mock_services, mock_settings):
    ctrl = JarvisControllerV2(
        config=mock_config,
        services=mock_services,
        settings=mock_settings,
    )
    assert ctrl.session_id is not None
    assert ctrl.config == mock_config
    assert ctrl.voice_enabled is False
    assert ctrl._gui_automation_enabled is False
    assert ctrl._app_launch_enabled is True

@pytest.mark.asyncio
async def test_controller_initialize_method(controller):
    await controller.startup()
    result = await controller.initialize()
    controller.memory.initialize.assert_awaited_once()
    assert result["session_id"] == controller.session_id
    assert result["memory_mode"] == "local"

@pytest.mark.asyncio
async def test_process_basic_flow(controller):
    response = await controller.process("hello jarvis")
    
    assert response == "LLM Response"
    controller.intent_router.route.assert_awaited_once()
    controller.llm_orchestrator.dispatch.assert_awaited_once()
    controller.profile.update_from_conversation.assert_called_once_with("hello jarvis", "LLM Response")
    
    # Check conversation buffer
    assert len(controller.memory_subsystem._conversation_buffer) == 1
    assert "User: hello jarvis\nJarvis: LLM Response" in controller.memory_subsystem._conversation_buffer[0]

@pytest.mark.asyncio
async def test_process_intent_routed(controller):
    # Mock intent router to return a response directly
    controller.intent_router.route.return_value = "Routed Response"
    
    response = await controller.process("what is my goal")
    
    assert response == "Routed Response"
    controller.intent_router.route.assert_awaited_once()
    controller.llm_orchestrator.dispatch.assert_not_called()
    controller.profile.update_from_conversation.assert_called_once_with("what is my goal", "Routed Response")

@pytest.mark.asyncio
async def test_process_long_input(controller):
    long_input = "a" * 5000
    await controller.process(long_input)
    
    # The input text should be truncated to 4000 characters before intent routing and dispatch
    # Since we mocked intent_router.route to return None, it goes to llm_dispatcher.dispatch
    called_text = controller.llm_orchestrator.dispatch.call_args[0][0]
    assert len(called_text) == 4000

@pytest.mark.asyncio
async def test_process_triggers_synthesis(controller):
    controller.synthesizer.should_run.return_value = True
    
    with patch.object(controller.memory_subsystem, "_schedule_synthesis") as mock_schedule:
        await controller.process("trigger synth")
        mock_schedule.assert_called_once()
        assert len(controller.memory_subsystem._conversation_buffer) == 0

@pytest.mark.asyncio
async def test_start_and_shutdown(controller):
    await controller.start()
    controller.monitor.start.assert_awaited_once()
    assert controller._goal_check_task is not None
    
    await controller.shutdown()
    controller.monitor.stop.assert_awaited_once()
    assert controller._goal_check_task.cancelled() or controller._goal_check_task.done()

@pytest.mark.asyncio
async def test_session_summary(controller):
    summary = controller.session_summary()
    assert summary["session_id"] == controller.session_id
    assert summary["exchanges"] == 0
    
    await controller.process("test")
    summary = controller.session_summary()
    assert summary["exchanges"] == 1

def test_disabled_messages(controller):
    assert "Desktop control is disabled" in controller._desktop_control_disabled_message()
    assert "Application launch is disabled" in controller._app_launch_disabled_message()

def test_looks_like_desktop_control_request(controller):
    assert controller._looks_like_desktop_control_request("please click the button") is True
    assert controller._looks_like_desktop_control_request("type hello world") is True
    assert controller._looks_like_desktop_control_request("what is the weather") is False

@pytest.mark.asyncio
async def test_handle_goal_intent(controller):
    with patch("core.controller_v2.handle_goal_intent") as mock_handler:
        mock_result = MagicMock()
        mock_result.mutated = True
        mock_result.response = "Goal updated"
        mock_handler.return_value = mock_result
        
        response = await controller._handle_goal_intent("add a goal", "add a goal")
        
        assert response == "Goal updated"
        controller.goal_runner.persist_goal_state.assert_called_once()

@pytest.mark.asyncio
async def test_handle_preference_intent(controller):
    with patch("core.controller_v2.handle_preference_intent", new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = "Preference saved"
        
        response = await controller._handle_preference_intent("i like blue", "i like blue")
        
        assert response == "Preference saved"
        mock_handler.assert_awaited_once_with("i like blue", "i like blue", memory=controller.memory)

@pytest.mark.asyncio
async def test_run_cli_exit(controller):
    # Test that run_cli exits when 'exit' is inputted
    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        # Simulate input returning 'exit'
        mock_loop.run_in_executor = AsyncMock(return_value="exit")
        
        await controller.run_cli()
        
        # It should exit gracefully
        mock_loop.run_in_executor.assert_awaited()

