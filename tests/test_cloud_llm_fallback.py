from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.llm.client import LLMClientV2
from core.llm.cloud_client import CloudLLMClient


def _make_response(status: int, payload: dict) -> AsyncMock:
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=payload)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


def _make_session(response: AsyncMock) -> MagicMock:
    session = MagicMock()
    session.post = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def test_cloud_llm_provider_priority():
    env = {
        "OPENAI_API_KEY": "openai-key",
        "GROQ_API_KEY": "groq-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
    }
    with patch.dict(os.environ, env, clear=True):
        client = CloudLLMClient()
    assert client._available == ["groq", "openai", "anthropic"]


@pytest.mark.asyncio
async def test_cloud_llm_skips_failed_provider():
    env = {"GROQ_API_KEY": "groq-key", "OPENAI_API_KEY": "openai-key"}
    with patch.dict(os.environ, env, clear=True):
        client = CloudLLMClient()
        client._call_groq = AsyncMock(side_effect=RuntimeError("Groq unavailable"))
        client._call_openai = AsyncMock(return_value="openai answer")

        result = await client.complete("hello", system="be terse", temperature=0.2)

    assert result == "openai answer"
    client._call_groq.assert_awaited_once_with("hello", "be terse", 0.2)
    client._call_openai.assert_awaited_once_with("hello", "be terse", 0.2)


@pytest.mark.asyncio
async def test_cloud_llm_raises_without_configured_providers():
    with patch.dict(os.environ, {}, clear=True):
        client = CloudLLMClient()
        with pytest.raises(RuntimeError, match="All cloud LLM providers failed or are unconfigured."):
            await client.complete("hello")


@pytest.mark.asyncio
async def test_llm_client_falls_back_to_cloud_when_ollama_returns_empty():
    env = {"CLOUD_LLM_FALLBACK_ENABLED": "true", "GROQ_API_KEY": "groq-key"}
    cloud_client = MagicMock()
    cloud_client.complete = AsyncMock(return_value="cloud answer")
    session = _make_session(_make_response(200, {"response": ""}))

    with patch.dict(os.environ, env, clear=True):
        with patch("core.llm.client.CloudLLMClient", return_value=cloud_client):
            with patch("aiohttp.ClientSession", return_value=session):
                client = LLMClientV2()
                result = await client.complete("hello", system="be terse")

    assert result == "cloud answer"
    cloud_client.complete.assert_awaited_once_with("hello", system="be terse", temperature=0.1)


@pytest.mark.asyncio
async def test_llm_client_keeps_ollama_result_when_present():
    env = {"CLOUD_LLM_FALLBACK_ENABLED": "true", "GROQ_API_KEY": "groq-key"}
    cloud_client = MagicMock()
    cloud_client.complete = AsyncMock(return_value="cloud answer")
    session = _make_session(_make_response(200, {"response": "local answer"}))

    with patch.dict(os.environ, env, clear=True):
        with patch("core.llm.client.CloudLLMClient", return_value=cloud_client):
            with patch("aiohttp.ClientSession", return_value=session):
                client = LLMClientV2()
                result = await client.complete("hello", system="be terse")

    assert result == "local answer"
    cloud_client.complete.assert_not_awaited()
