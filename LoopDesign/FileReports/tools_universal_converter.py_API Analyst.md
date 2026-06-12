# API Analyst Report: tools\universal_converter.py

## Dependencies
- `import sys`
- `import subprocess`
- `import csv`
- `import json`
- `from pathlib import Path`

## Functions & Endpoints

### `ensure_package`
`def ensure_package(package_name, import_name=None)`
### `convert_image`
`def convert_image(src: Path, dst: Path)`
### `csv_to_json`
`def csv_to_json(src: Path, dst: Path)`
### `json_to_csv`
`def json_to_csv(src: Path, dst: Path)`
### `json_to_yaml`
`def json_to_yaml(src: Path, dst: Path)`
### `yaml_to_json`
`def yaml_to_json(src: Path, dst: Path)`
### `md_to_html`
`def md_to_html(src: Path, dst: Path)`
### `txt_to_html`
`def txt_to_html(src: Path, dst: Path)`
### `txt_to_pdf`
`def txt_to_pdf(src: Path, dst: Path)`
### `pdf_to_txt`
`def pdf_to_txt(src: Path, dst: Path)`
### `data_to_excel`
`def data_to_excel(src: Path, dst: Path)`
### `convert_media_ffmpeg`
`def convert_media_ffmpeg(src: Path, dst: Path)`
### `perform_conversion`
`def perform_conversion(source_path: str, target_format: str, output_path: str | None=None) -> str`
> Main entry point to convert files.
:param source_path: Path of the input file.
:param target_format: Target format extension (e.g. 'webp', 'pdf', 'csv', 'json').
:param output_path: Optional custom output file path.
:return: Path of the converted file.
