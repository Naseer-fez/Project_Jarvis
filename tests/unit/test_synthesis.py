import pytest
from unittest.mock import MagicMock, AsyncMock
from core.synthesis import ProfileSynthesizer


class MockProfile:
    def __init__(self, interaction_count):
        self.interaction_count = interaction_count
        self.applied_deltas = []

    def apply_delta(self, delta):
        self.applied_deltas.append(delta)
        return list(delta.keys())


def test_synthesizer_should_run():
    """Verify synthesizer should_run logic checks threshold counts."""
    synth = ProfileSynthesizer(llm=MagicMock())

    # Less than 20
    assert synth.should_run(MockProfile(19)) is False

    # Exactly 20
    assert synth.should_run(MockProfile(20)) is True

    # 21 (not divisible by 20)
    assert synth.should_run(MockProfile(21)) is False

    # 40 (divisible by 20)
    assert synth.should_run(MockProfile(40)) is True

    # Invalid profile
    assert synth.should_run(None) is False


@pytest.mark.asyncio
async def test_synthesize_success_sync_llm():
    """Verify synthesize processes sync LLM response and applies delta."""
    mock_llm = MagicMock()
    mock_llm.complete = MagicMock(return_value='{"name": {"value": "Alice", "confidence": 0.9}}')

    synth = ProfileSynthesizer(mock_llm)
    profile = MockProfile(20)

    result = await synth.synthesize(["user: hi", "jarvis: hello"], profile)

    assert result["updated_fields"] == ["name"]
    assert result["delta"] == {"name": {"value": "Alice", "confidence": 0.9}}
    assert profile.applied_deltas == [{"name": {"value": "Alice", "confidence": 0.9}}]


@pytest.mark.asyncio
async def test_synthesize_success_async_llm():
    """Verify synthesize handles awaitable LLM response."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value='<think>some reasoning</think>```json\n{"name": "Alice"}\n```')

    synth = ProfileSynthesizer(mock_llm)
    profile = MockProfile(20)

    result = await synth.synthesize(["user: hi", "jarvis: hello"], profile)

    assert result["updated_fields"] == ["name"]
    assert result["delta"] == {"name": "Alice"}


@pytest.mark.asyncio
async def test_synthesize_invalid_json():
    """Verify synthesis handles invalid JSON response gracefully."""
    mock_llm = MagicMock()
    mock_llm.complete = MagicMock(return_value='not a json string')

    synth = ProfileSynthesizer(mock_llm)
    profile = MockProfile(20)

    result = await synth.synthesize(["user: hi"], profile)
    assert result["error"] == "invalid_json"
    assert result["updated_fields"] == []


@pytest.mark.asyncio
async def test_synthesize_llm_unavailable():
    """Verify synthesis handles missing LLM response gracefully."""
    # LLM without complete attribute
    synth = ProfileSynthesizer(object())
    profile = MockProfile(20)

    result = await synth.synthesize(["user: hi"], profile)
    assert result["error"] == "llm_unavailable"
    assert result["updated_fields"] == []
