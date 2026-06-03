from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
import aiohttp
import pytest

from core.llm.ollama_client import OllamaClient


class MockResponse:
    def __init__(self, status: int, json_data: dict, text_data: str = "") -> None:
        self.status = status
        self._json_data = json_data
        self._text_data = text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


@pytest.mark.asyncio
async def test_ollama_client_success():
    client = OllamaClient(base_url="http://localhost:11434")
    
    mock_resp = MockResponse(200, {"response": "Hello world!"})
    
    with patch("aiohttp.ClientSession.post", return_value=mock_resp) as mock_post:
        result = await client.complete("test prompt", keep_think=True)
        assert result == "Hello world!"
        mock_post.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)  # Mock sleep to run fast
async def test_ollama_client_retry_on_client_error(mock_sleep):
    client = OllamaClient(base_url="http://localhost:11434")
    
    mock_resp = MockResponse(200, {"response": "Success after error"})
    
    # 1st call raises aiohttp.ClientError, 2nd call returns success
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = [
            aiohttp.ClientError("connection reset"),
            mock_resp
        ]
        
        result = await client.complete("test prompt", keep_think=True)
        assert result == "Success after error"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.5)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_ollama_client_retry_on_timeout(mock_sleep):
    client = OllamaClient(base_url="http://localhost:11434")
    
    mock_resp = MockResponse(200, {"response": "Success after timeout"})
    
    # 1st call raises asyncio.TimeoutError, 2nd call returns success
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = [
            asyncio.TimeoutError("request timeout"),
            mock_resp
        ]
        
        result = await client.complete("test prompt", keep_think=True)
        assert result == "Success after timeout"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.5)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_ollama_client_retry_on_transient_http_status(mock_sleep):
    client = OllamaClient(base_url="http://localhost:11434")
    
    mock_resp_503 = MockResponse(503, {}, "Service Unavailable")
    mock_resp_success = MockResponse(200, {"response": "Success after 503"})
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = [
            mock_resp_503,
            mock_resp_success
        ]
        
        result = await client.complete("test prompt", keep_think=True)
        assert result == "Success after 503"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.5)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_ollama_client_no_retry_on_permanent_http_status(mock_sleep):
    client = OllamaClient(base_url="http://localhost:11434")
    
    mock_resp_404 = MockResponse(404, {}, "Model not found")
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value = mock_resp_404
        
        with pytest.raises(RuntimeError, match="Ollama HTTP 404"):
            await client.complete("test prompt", keep_think=True)
            
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_ollama_client_no_retry_on_empty_response(mock_sleep):
    client = OllamaClient(base_url="http://localhost:11434")
    
    mock_resp_empty = MockResponse(200, {"response": "   "})
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value = mock_resp_empty
        
        with pytest.raises(RuntimeError, match="Ollama returned empty response"):
            await client.complete("test prompt", keep_think=True)
            
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_ollama_client_exhausts_retries(mock_sleep):
    client = OllamaClient(base_url="http://localhost:11434")
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.side_effect = aiohttp.ClientError("network down")
        
        with pytest.raises(aiohttp.ClientError, match="network down"):
            await client.complete("test prompt", keep_think=True)
            
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
