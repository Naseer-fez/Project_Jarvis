# Analysis Report for payload_extractor.py

## Dependencies
- pathlib.Path
- re

## Schemas
- PayloadExtractor

## API Contracts
- _normalize_text(value)
- _truncate(text, max_chars)
- read_text_file(path, max_bytes)
- PayloadExtractor.__init__(self, max_text_chars_per_item, video_frame_interval_seconds, max_video_samples)
- PayloadExtractor.extract_text_payload(self, path)
- PayloadExtractor.extract_text_from_image(self, path)
- PayloadExtractor.extract_text_from_video(self, path)

## Configuration Variables
- _TEXT_EXTENSIONS
- _IMAGE_EXTENSIONS
- _VIDEO_EXTENSIONS

## Assumptions & Notes
None

