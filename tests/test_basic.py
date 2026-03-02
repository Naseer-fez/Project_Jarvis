"""
tests/test_basic.py — Basic smoke test for optional whisper dependency.
Skipped automatically when whisper is not installed.
"""
import pytest

whisper = pytest.importorskip("whisper", reason="openai-whisper not installed — skipping")


def test_whisper_loads_small_model():
    model = whisper.load_model("small")
    assert model is not None
