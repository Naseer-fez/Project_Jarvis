# API Analyst Report: automation\payload_extractor.py

## Dependencies
- `from pathlib import Path`
- `import re`

## Configuration Variables
- `_TEXT_EXTENSIONS` = `{'.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.csv', '.tsv', '.py', '.js', '.ts', '.html', '.css', '.ini', '.log'}`
- `_IMAGE_EXTENSIONS` = `{'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}`
- `_VIDEO_EXTENSIONS` = `{'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}`

## Schemas & API Contracts (Classes)

### Class `PayloadExtractor`
**Methods:**
- `def __init__(self, max_text_chars_per_item: int, video_frame_interval_seconds: float, max_video_samples: int)`
- `def extract_text_payload(self, path: Path) -> str`
- `def extract_text_from_image(self, path: Path) -> str`
- `def extract_text_from_video(self, path: Path) -> str`


## Functions & Endpoints

### `_normalize_text`
`def _normalize_text(value: str) -> str`
### `_truncate`
`def _truncate(text: str, max_chars: int) -> str`
### `read_text_file`
`def read_text_file(path: Path, max_bytes: int=2000000) -> str`