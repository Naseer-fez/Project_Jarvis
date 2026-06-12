# Data Model Analyst Report: en_US-lessac-medium.onnx.json

## File Analysis
- **Filename**: `en_US-lessac-medium.onnx.json`
- **Path**: `d:\AI\Jarvis\data\voices\en_US-lessac-medium.onnx.json`
- **Format**: JSON Document

## Schema and State Objects
This file is the configuration descriptor for the adjacent Piper TTS ONNX model (`en_US-lessac-medium.onnx`).

### Key Schema Fields
- **`audio`**: Configures sample rate (22050) and quality ("medium").
- **`espeak`**: Specifies the base espeak-ng voice (`en-us`).
- **`inference`**: Tuning parameters (`noise_scale`: 0.667, `length_scale`: 1, `noise_w`: 0.8).
- **`phoneme_type`**: "espeak" (indicates phonemization backend).
- **`phoneme_id_map`**: A massive dictionary mapping espeak phoneme characters (like `"A"`, `","`, etc.) to discrete integer IDs required by the ONNX model inputs.
- **`num_symbols`**: 256
- **`num_speakers`**: 1
- **`piper_version`**: "1.0.0"
- **`language` & `dataset`**: Metadata marking it as US English, lessac dataset.

## Assumptions & Contracts
- The TTS engine using this model strictly requires Piper-compatible initialization and matching phoneme mappings.
- The `inference` defaults must be parsed and passed to the ONNX runtime or Piper wrapper.

## Dependencies & Variables
- Tightly coupled with the `en_US-lessac-medium.onnx` weights file.
- Relies on the `espeak-ng` system library/binary for text-to-phoneme pre-processing.

## Extracted Prompts
None found.
