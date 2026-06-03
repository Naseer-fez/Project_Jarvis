import pytest
import configparser
from unittest.mock import AsyncMock, patch
from core.controller_v2 import JarvisControllerV2
from core.controller.web_search import handle_web_search

@pytest.mark.asyncio
async def test_offline_fallback_with_preference(tmp_path):
    """Verify that when the LLM is offline, controller falls back to stored preferences in memory."""
    config = configparser.ConfigParser()
    config.add_section("memory")
    db_file = tmp_path / "test_memory.db"
    config.set("memory", "db_path", str(db_file))

    controller = JarvisControllerV2(config=config, voice=False)
    await controller.initialize()
    await controller.start()

    # Store a preference in memory
    await controller.memory.store_preference("favorite color", "blue")

    # Mock the LLM to return empty string (simulating offline/failure)
    controller.llm.chat_async = AsyncMock(return_value="")

    # Ask the controller about the preference
    response = await controller.process("favorite color")

    assert "Offline fallback from memory: blue" in response

    await controller.shutdown()


@pytest.mark.asyncio
async def test_offline_fallback_no_preference(tmp_path):
    """Verify that when LLM is offline and no preference is found, a generic offline message is returned."""
    config = configparser.ConfigParser()
    config.add_section("memory")
    db_file = tmp_path / "test_memory.db"
    config.set("memory", "db_path", str(db_file))

    controller = JarvisControllerV2(config=config, voice=False)
    await controller.initialize()
    await controller.start()

    # Mock the LLM to return empty string
    controller.llm.chat_async = AsyncMock(return_value="")

    # Ask something not in preferences
    response = await controller.process("what is the speed of light")

    assert response == "I don't know while offline."

    await controller.shutdown()


@pytest.mark.asyncio
async def test_web_search_offline_fallback(tmp_path):
    """Verify that when web search or its synthesis LLM call fails, the system falls back to memory or offline message."""
    config = configparser.ConfigParser()
    config.add_section("memory")
    db_file = tmp_path / "test_memory.db"
    config.set("memory", "db_path", str(db_file))

    controller = JarvisControllerV2(config=config, voice=False)
    await controller.initialize()
    await controller.start()

    # Store a preference in memory
    await controller.memory.store_preference("weather in paris", "always sunny")

    # Mock the LLM to return empty string
    controller.llm.chat_async = AsyncMock(return_value="")

    # Mock the web search tool to raise an exception to force the LLM fallback path
    with patch("core.tools.web_tools.web_search", side_effect=Exception("network down")):
        # Test handle_web_search directly to verify its fallback logic.
        response = await handle_web_search(
            user_input="weather in paris",
            trace_id="test-trace",
            memory=controller.memory,
            llm=controller.llm,
            model_router=controller.model_router,
            profile=controller.profile
        )

        assert "Offline fallback from memory: always sunny" in response

        # Test when no preference matches
        response_no_pref = await handle_web_search(
            user_input="what is the population of mars",
            trace_id="test-trace",
            memory=controller.memory,
            llm=controller.llm,
            model_router=controller.model_router,
            profile=controller.profile
        )
        assert response_no_pref == "I don't know while offline."

    await controller.shutdown()
