# Configuration Analyst Report: en_US-lessac-medium.onnx.json

## File Overview
- **Path**: `d:\AI\Jarvis\data\voices\en_US-lessac-medium.onnx.json`
- **Type**: JSON Configuration
- **Purpose**: Defines phonetic, acoustic, and environmental properties for a Piper TTS model deployment.

## Exhaustive Line-by-Line / Schema Analysis
1. **Audio Configuration**:
   - `sample_rate`: `22050` Hz implicitly required for audio playback.
   - `quality`: `"medium"`.
2. **ESpeak Configuration**:
   - `voice`: `"en-us"`. Relies on eSpeak-ng backend for phonemization.
3. **Inference Parameters**:
   - `noise_scale: 0.667`, `length_scale: 1`, `noise_w: 0.8`. Constants used for the VITS model generator tuning.
4. **Phoneme Maps** (`phoneme_id_map`):
   - Extensive explicit mapping of IPA characters (e.g., `_`, `a`, `b`, `æ`, `ɹ`) to integer tensor IDs for the ONNX inference graph.
5. **Model Metadata**:
   - `num_symbols`: 256
   - `num_speakers`: 1 (Single-speaker model).
   - `piper_version`: `"1.0.0"`. **API Contract**: Expects Piper text-to-speech runtime at roughly v1.0.0 APIs.
   - `language`: Code (`en_US`), Family (`en`), Region (`US`), Name (`English`).
   - `dataset`: `"lessac"`.

## Implicit Environment Assumptions
- **Dependency**: The execution environment strictly requires `espeak-ng` installed or bundled for text-to-phoneme conversion.
- **Execution Target**: Expects ONNX Runtime (`onnxruntime`) to read the adjacent `.onnx` model binary.

## Secrets & Env Vars
- Pure configuration file. No secrets, credentials, or API tokens.
