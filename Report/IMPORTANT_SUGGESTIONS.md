# Important Suggestions

Generated: 2026-05-30

These suggestions are prioritized by maintenance and safety impact.

## 1. Move generated/runtime artifacts out of source control

High priority.

The repository contains runtime and generated folders such as `jarvis_env`, `htmlcov`, `.coverage`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `logs`, `outputs`, `data/chroma`, and Chroma DB folders.

Suggested action:

- Confirm these are ignored in `.gitignore`.
- Keep only curated sample data in the repository.
- Store local databases, coverage reports, and runtime outputs outside committed source.

Why it matters:

- Reduces repo noise.
- Prevents accidental commits of large or private runtime data.
- Makes future LLM/code review much less confusing.

## 2. Decide what to do with duplicate and legacy folders

High priority.

Folders such as `archive_legacy`, `Jarvis-2.0`, and `to_commit_pool` contain old or duplicate project material. They are useful as references, but they can confuse imports, search results, and future agents.

Suggested action:

- Move them outside the active repo, or document them as non-runtime references.
- If they must stay, exclude them from lint, tests, coverage, and LLM workspace summaries.

## 3. Restrict dashboard file viewing/downloading

High priority security review.

`dashboard/server.py` has an authenticated file-view route. It resolves a supplied path and returns the file if it exists. Authentication helps, but a safer dashboard should restrict file reads to approved project/output directories.

Suggested action:

- Add an allowlist such as project root, `outputs`, `workspace`, and selected report folders.
- Reject paths outside those roots.
- Add tests for path traversal and absolute paths outside the workspace.

## 4. Migrate dashboard lifecycle hooks

Medium priority.

FastAPI reports deprecation warnings for `@app.on_event("startup")` and `@app.on_event("shutdown")`.

Suggested action:

- Replace them with a FastAPI lifespan context manager.
- Keep tests around dashboard startup/shutdown behavior.

## 5. Update Starlette template and cookie test patterns

Medium priority.

Warnings remain for:

- `TemplateResponse(name, {"request": request})`
- per-request cookies in tests

Suggested action:

- Use `TemplateResponse(request, name, context)`.
- Set cookies on the test client directly instead of per request.

## 6. Add a fast default test profile

Medium priority.

The normal `./run-tests.ps1` uses coverage by default and timed out in a short 2-minute run. The fast no-coverage suite passes but takes around 3.6 minutes.

Suggested action:

- Add a script such as `run-fast-tests.ps1` for `pytest -q --no-cov`.
- Keep full coverage in CI or a separate command.
- Split slow/integration tests with markers.

## 7. Pin dependency versions more tightly

Medium priority.

Requirement files currently use broad package names. This can cause surprise deprecations or breakages when dependencies update.

Suggested action:

- Pin or constrain major versions for FastAPI, Pydantic, Starlette, pytest, chromadb, sentence-transformers, and model/provider SDKs.
- Maintain a lock file for reproducible local setup.

## 8. Repair README encoding artifacts if they exist in the file

Low to medium priority.

The terminal displayed mojibake-like characters in the README diagrams. This may be a console encoding issue, but it is worth verifying the file in an editor.

Suggested action:

- Confirm the README is UTF-8.
- Replace fragile box-drawing diagrams with Mermaid or plain ASCII if needed.

## 9. Avoid starting duplicate background monitor tasks

Low to medium priority.

`JarvisControllerV2.start()` and `JarvisControllerV2.run_cli()` both schedule monitor start tasks. Tests pass, but this deserves review.

Suggested action:

- Make monitor startup idempotent, or ensure it starts in only one runtime phase.
- Add a test that monitor startup is not duplicated.

## 10. Keep the public entrypoint compatibility contract documented

Low priority.

`main.py` now intentionally re-exports launcher compatibility names. Future cleanup should preserve that contract or update tests and migration notes at the same time.

Suggested action:

- Keep `main.py.__all__` synchronized with intended public launcher imports.
- Treat `main_connector.py` as the implementation surface and `main.py` as the compatibility launcher.

