# core/voice/modeling_vibevoice_acoustic_tokenizer.py
# Experimental voice tokenizer — DISABLED by default.
# Guards all code with try/except so it never breaks the import chain.

try:
    raise ImportError(
        "VibeVoice acoustic tokenizer is an experimental module and is currently "
        "disabled. It is not required for normal operation. To enable it, remove "
        "the ImportError guard and ensure all optional dependencies are installed."
    )

    # ---------------------------------------------------------------------------
    # Placeholder implementation (unreachable until guard is removed)
    # ---------------------------------------------------------------------------

    class VibeVoiceAcousticTokenizer:  # noqa: F811
        """Stub tokenizer class — not functional until module is enabled."""

        def __init__(self, config=None):
            self.config = config or {}

        def tokenize(self, audio):
            raise NotImplementedError("Acoustic tokenizer not implemented.")

        def decode(self, tokens):
            raise NotImplementedError("Acoustic tokenizer not implemented.")

except ImportError:
    # Controlled failure — surface a clean placeholder so callers can
    # detect unavailability without a hard crash.

    class VibeVoiceAcousticTokenizer:  # type: ignore[no-redef]
        """
        Unavailable stub — acoustic tokenizer module is disabled.
        Instantiating this class will raise ImportError.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "VibeVoiceAcousticTokenizer is not available. "
                "The experimental module has been disabled."
            )

except Exception as e:
    import warnings

    warnings.warn(
        f"Unexpected error loading VibeVoiceAcousticTokenizer: {e}",
        RuntimeWarning,
        stacklevel=1,
    )

    class VibeVoiceAcousticTokenizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"VibeVoiceAcousticTokenizer failed to load: {e}")


__all__ = ["VibeVoiceAcousticTokenizer"]
