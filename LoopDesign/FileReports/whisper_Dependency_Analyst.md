# Dependency Analysis: Whisper STT

## Overview
`small.pt` located in the `whisper` directory.

## Schemas / API Contracts
- Standard PyTorch saved model format (`.pt`).

## Assumptions & Dependencies
- Hard dependency on **OpenAI Whisper** or a compatible library capable of loading Whisper PyTorch weights for Speech-to-Text inference.
- Hard dependency on **PyTorch** (`torch`) framework.
- Assumes sufficient local compute/memory to load and run the "small" parameter scale Whisper model.
