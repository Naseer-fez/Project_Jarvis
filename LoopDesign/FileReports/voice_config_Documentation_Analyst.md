# Documentation Analysis: `voices/en_US-lessac-medium.onnx.json`

## Target
`d:\AI\Jarvis\data\voices\en_US-lessac-medium.onnx.json`

## Overview
Configuration file for the local TTS (Text-To-Speech) model. This specifies inference variables, phoneme mappings, and metadata for Piper TTS.

## Configuration Variables & API Contracts
- **Audio Output**: `sample_rate`: 22050 Hz, `quality`: "medium"
- **Backend Model**: Uses `espeak` with `voice: "en-us"`.
- **Inference Variables**:
  - `noise_scale`: 0.667
  - `length_scale`: 1
  - `noise_w`: 0.8
- **Piper TTS Version**: "1.0.0"
- **Dataset**: "lessac"
- **Phoneme Map**: An extensive mapping from textual characters and IPA phonemes to integer IDs used by the ONNX model.

## Developer Notes
- JARVIS utilizes Piper TTS for local, low-latency offline voice synthesis.
- To modify the default speaking rate, `length_scale` can be adjusted.
- The presence of this model matches the `main_v3.py` logs highlighting the "Voice Layer" startup.
