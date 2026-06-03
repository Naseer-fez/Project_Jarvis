import configparser
import pytest
from core.runtime.container import ServiceContainer
from core.controller.services import build_controller_services


class MockMemory:

    def __init__(self, db_path, chroma_path=None, model_name=None):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.custom_flag = True

    def set_llm(self, llm, enable_context_titles=True):
        self.llm = llm


def test_service_container_basic_flow():
    container = ServiceContainer()

    # Test class registration
    container.register("memory", MockMemory)
    assert container.has("memory")

    # Resolve
    mem = container.resolve("memory", db_path="test.db")
    assert isinstance(mem, MockMemory)
    assert mem.db_path == "test.db"
    assert mem.custom_flag

    # Test instance registration
    class Dummy:
        pass

    dummy_inst = Dummy()
    container.register_instance("dummy", dummy_inst)
    assert container.resolve("dummy") is dummy_inst


def test_build_controller_services_respects_container_overrides():
    config = configparser.ConfigParser()
    config.add_section("memory")
    config.set("memory", "db_path", "test_memory.db")
    config.add_section("ollama")
    config.set("ollama", "base_url", "http://localhost:11434")

    container = ServiceContainer()
    # Override the "memory" service before building
    container.register("memory", MockMemory)

    # Build services using the pre-configured container
    settings, services = build_controller_services(config, container=container)

    # Assert container is exposed on services
    assert services.container is container

    # Assert that the custom memory mock was resolved instead of HybridMemory
    assert isinstance(services.memory, MockMemory)
    assert services.memory.custom_flag
    assert services.memory.db_path == settings.db_path


def test_service_container_edge_cases():
    container = ServiceContainer()

    # 1. Unregistered resolution raises ValueError
    with pytest.raises(ValueError, match="is not registered"):
        container.resolve("non_existent_service")

    # 2. Re-registration invalidates singleton cache
    class ServiceA:
        def __init__(self, val=1):
            self.val = val

    container.register("service_a", ServiceA, is_singleton=True)
    inst1 = container.resolve("service_a", val=10)
    assert inst1.val == 10

    # Resolve again (returns cached singleton)
    inst2 = container.resolve("service_a", val=20)
    assert inst2 is inst1
    assert inst2.val == 10

    # Re-register
    container.register("service_a", ServiceA, is_singleton=True)
    # Should resolve a new instance now since cache was invalidated
    inst3 = container.resolve("service_a", val=30)
    assert inst3 is not inst1
    assert inst3.val == 30

    # 3. Factory function registration
    def my_factory():
        return "factory_output"

    container.register("factory_service", my_factory, is_singleton=False)
    assert container.resolve("factory_service") == "factory_output"

    # 4. Check container reset clears registrations
    container.reset()
    assert not container.has("service_a")
    with pytest.raises(ValueError):
        container.resolve("service_a")
