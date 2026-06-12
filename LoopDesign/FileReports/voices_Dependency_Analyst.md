# Dependency Analysis: Voices (Piper TTS)

## Overview
Files `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json` in the `voices` directory.

## Schemas / API Contracts
- The `.json` file schema dictates Piper TTS configuration settings.
- Maps `espeak` phonemes to unique tensor IDs (0-256) for the ONNX model.

## Assumptions & Dependencies
- Hard dependency on **Piper TTS** library (`"piper_version": "1.0.0"`).
- Hard dependency on **ONNX runtime** to execute the `.onnx` acoustic model.
- Hard dependency on **espeak / espeak-ng** as the phonemizer backend (`"phoneme_type": "espeak"`, `"voice": "en-us"`).
- Configuration variables:
  - `sample_rate`: 22050
  - `quality`: medium
  - `noise_scale`: 0.667
  - `length_scale`: 1
  - `noise_w`: 0.8
  - `language`: `en_US` (English, United States)
  - `dataset`: `lessac`
