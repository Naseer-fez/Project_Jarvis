# API Analyst Report: voice\audio_playback.py

## Dependencies
- `import shutil`
- `import subprocess`
- `from typing import Optional`

## Schemas & API Contracts (Classes)

### Class `AudioPlayer`
> Plays raw audio using ffplay.

**Methods:**
- `def __init__(self, sample_rate: int) -> None`
  - *Initializes audio player.*
- `def __enter__(self)`
  - *Starts ffplay subprocess and returns player.*
- `def __exit__(self, exc_type, exc_val, exc_tb)`
  - *Stops ffplay subprocess.*
- `def play(self, audio_bytes: bytes) -> None`
  - *Plays raw audio using ffplay.*
- @staticmethod
- `def is_available() -> bool`
  - *Returns true if ffplay is available.*

