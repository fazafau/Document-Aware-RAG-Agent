# AGENTS

## Python Coding Standards

- Prefer small, single-purpose functions.
- Use explicit type hints on public functions and data structures.
- Use `pathlib.Path` for filesystem paths instead of raw strings where practical.
- Validate inputs early and raise specific exceptions.
- Keep ordering deterministic when processing files or documents.
- Favor readable standard-library code before adding abstraction.
- Use existing project libraries only when they improve clarity or reduce risk.
- Keep constants near the top of the module.
- Keep comments short and factual.
- Preserve module and function docstrings.

## Commenting Standard

- Write a short comment for almost every line or small cluster of lines you add.
- Explain intent, not syntax.
- Keep comments concrete and brief.
- Keep the existing top-of-file and function-level docstrings in place.

## Repo Conventions

- Build RAG ingestion as a sequence: load, clean, split, then hand off to embeddings or retrieval.
- Preserve metadata on every `Document` that should survive into later pipeline stages.
- Prefer deterministic behavior for document indexing and chunk ordering.
- Keep new code aligned with the current repo structure in `app/`, `data/`, and `skills/`.

## Editing Guidance

- Start with `app/ingest.py` when introducing new Python style conventions.
- Apply the same standard to later files as they are created or updated.
