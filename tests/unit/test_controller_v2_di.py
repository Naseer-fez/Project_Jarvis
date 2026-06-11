import pytest
import configparser
from core.controller_v2 import JarvisControllerV2
from core.runtime.container import ServiceContainer


class MockMemory:
    def __init__(self, *args, **kwargs):
        self.mode = "mock"
    async def initialize(self, *args, **kwargs):
        return {"mode": self.mode}
    async def close(self):
        pass


class MockModelRouter:
    def get_best_available(self, *args, **kwargs):
        return "mock-model"


def test_controller_v2_di_override():
    container = ServiceContainer()
    container.register("memory", MockMemory)
    
    config = configparser.ConfigParser()
    config.add_section("memory")
    
    controller = JarvisControllerV2(config=config, container=container)
    
    # Assert the controller grabbed our mock memory
    assert isinstance(controller.memory, MockMemory)
    assert controller.memory.mode == "mock"
    
    # Assert standard components are still injected from the fallback builder
    assert controller.tool_router is not None
    assert controller.agent_loop is not None


@pytest.mark.asyncio
async def test_controller_v2_initialize_flow():
    container = ServiceContainer()
    container.register("memory", MockMemory)
    controller = JarvisControllerV2(container=container)
    
    await controller.startup()
    res = await controller.initialize()
    assert res["memory_mode"] == "mock"
    assert "session_id" in res

