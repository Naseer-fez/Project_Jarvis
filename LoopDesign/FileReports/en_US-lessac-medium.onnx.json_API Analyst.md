# API Analyst Report: en_US-lessac-medium.onnx.json

## Overview
This file contains the JSON configuration and API mapping for a Piper Text-to-Speech (TTS) ONNX model voice. It specifies the properties, phoneme mapping schema, and audio format contracts.

## Schema / Structure
The file structure assumes the following root contract:
- `audio`: Object defining audio rendering properties (sample rate, quality).
- `espeak`: Object for the eSpeak backend configuration.
- `inference`: Object mapping model inference weights.
- `phoneme_type`: String describing the phonetic mapping standard used.
- `phoneme_map` / `phoneme_id_map`: Dictionary mapping characters to integer arrays for ONNX model consumption.
- `num_symbols` / `num_speakers`: Integer constraints.
- `speaker_id_map`: Empty object (suggests single speaker).
- `piper_version`: Semantic version constraint.
- `language`: Metadata regarding the language family and locale.
- `dataset`: String identifier for the model dataset source.

## Line-by-Line Analysis
- **Lines 1-5**: `audio` object. Contracts `sample_rate` to `22050` Hz, `quality` to `"medium"`. External playback APIs must respect these exact format parameters.
- **Lines 6-8**: `espeak` object maps `voice` to `"en-us"`. External phonemizer dependency.
- **Lines 9-13**: `inference` object configures `noise_scale` (`0.667`), `length_scale` (`1`), `noise_w` (`0.8`). These are ML inference hyperparameters for the ONNX backend.
- **Line 14**: `phoneme_type` explicitly set to `"espeak"`.
- **Lines 15-479**: Phonetic dictionary mapping individual symbols/phonemes to ONNX embedding IDs (0-153). This acts as a strict API contract between text processing and the neural network input layer.
- **Line 480**: `num_symbols` set to `256`. The neural network architecture expects exactly this dimension.
- **Line 481-482**: `num_speakers` set to `1`, `speaker_id_map` is empty, confirming a single-speaker ONNX setup.
- **Line 483**: `piper_version` constraint `1.0.0`. Implies dependency on piper-tts.
- **Lines 484-491**: `language` object provides explicit strings for `en_US`. Useful for localization APIs.
- **Line 492**: `dataset` is `"lessac"`.

## Assumptions & API Contracts
- **Piper TTS**: The entire JSON is an API payload tailored for the `piper-tts` engine or compatible ONNX runners.
- **Audio Consumer**: Downstream audio APIs (e.g., PyAudio, ffmpeg) must be initialized at 22050Hz for playback of outputs generated via this config.

## Dependencies
- Dependencies on `eSpeak NG` for phonetic translation.
- Dependency on the ONNX runtime for `en_US-lessac-medium.onnx` execution.
