# File Report: automation_ingest.jsonl
Role: Runtime Investigator

## Overview
This file operates as an append-only transaction log or event stream for the Retrieval-Augmented Generation (RAG) ingestion pipeline. It tracks every document (specifically screenshots) successfully processed by the system.

## Schemas & Contracts
The file follows the JSON Lines (JSONL) format, where each line is a valid JSON object.
Schema per line:
- `timestamp` (String): ISO-8601 timestamp with timezone of the ingestion event. Example: `"2026-05-25T16:32:46.725322+00:00"`.
- `type` (String): Event type, hardcoded as `"rag_ingest"`.
- `path` (String): Absolute filesystem path to the ingested source file. Example: `"D:\\AI\\Jarvis\\outputs\\screenshots\\20260525_220238.png"`.
- `source` (String): Identifies the source category, seen exclusively as `"screenshot"`.
- `chars` (Integer): Total number of characters extracted (likely via OCR) from the image.
- `chunks` (Integer): The number of text chunks the extracted text was split into for embedding generation.

## Assumptions & API Contracts
- The system assumes chronological ordering of lines based on append operations.
- The pipeline processes screenshots from `D:\AI\Jarvis\outputs\screenshots\`.
- An external OCR or vision component extracts text from the images, returning a character count.
- A text splitter configuration is applied to break the text into `chunks` before generating embeddings and storing them in a vector database.

## Execution Paths & State Transitions
- **Main Loop (Ingestion Pipeline)**: 
  1. Detect new screenshot in the outputs directory.
  2. Read image, process via OCR (yielding text).
  3. Chunk the text into a number of partitions.
  4. Vectorize and store chunks in the memory database.
  5. Append a log entry to `automation_ingest.jsonl`.
- **State Transition**: `Unprocessed File -> Extracted Text -> Chunked Text -> Indexed Vector -> Logged Event`.

## Dependencies & Variables
- Dependencies: RAG ingestion module, OCR/Vision API, text splitter logic.
- Target Input Directory: `D:\AI\Jarvis\outputs\screenshots\`

## Prompts Found
None.
