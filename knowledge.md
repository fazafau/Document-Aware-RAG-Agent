# Knowledge

## Decisions
- Ingestion uses LangChain `Document` and `RecursiveCharacterTextSplitter`, but PDF extraction is handled directly with `pypdf` for explicit control over page metadata and cleanup.
- The knowledge-capture skill lives in `skills/knowledge-capture/` and writes durable notes into this `knowledge.md` file.
- Skill metadata belongs in `skills/<skill-name>/agents/openai.yaml`, not in a repo-root `.agents` file.

## Concepts
- LangChain is useful as a set of composable building blocks, not as a requirement to use every loader or pipeline stage.
- For RAG ingestion, the important concepts are document loading, normalization, chunking, metadata preservation, embeddings, vector storage, and retrieval.
- A narrow loader that extracts text predictably can be better than a fully abstracted loader when the goal is repeatable indexing and clear failure modes.
- Conceptual notes should capture the mental model needed to build later, not just the commands or file layout.
- LangChain's `Document` is the standard text-plus-metadata container used between ingestion, splitting, embeddings, and retrieval.
- `RecursiveCharacterTextSplitter` is the chunking utility responsible for splitting text while trying to preserve natural boundaries.
- `pypdf.PdfReader` gives direct control over PDF parsing and text extraction, which makes cleanup and page metadata handling explicit.
- In `app/ingest.py`, the pipeline is intentionally staged: normalize paths, read PDFs, clean page text, build `Document` objects, then split them into chunks.
- `load_documents()` is the main loader because it returns page-level `Document` objects with metadata already attached.
- `split_documents()` is the second stage because chunking should happen after page metadata and cleaned text exist.
- `ingest_documents()` is a convenience wrapper that combines loading and splitting into one call for downstream code.
- The CLI in `app/ingest.py` is only a smoke test and preview tool; it does not build embeddings or a vector store.

## Python
- `Path` from `pathlib` is a filesystem path object, not just a string.
- `str`, `list`, and `Document` are classes in Python; values like `"text"` or a `Document(...)` result are instances of those classes.
- `Iterable` is usually a typing concept that means "something you can loop over", not a concrete object you create for its own sake.
- `from __future__ import annotations` delays type-hint evaluation, which makes annotation-heavy code easier to write and maintain.
- A function annotated as `-> list[Document]` communicates the exact shape of the returned data.
- Raising `FileNotFoundError` and `ValueError` early makes ingestion failures easier to diagnose.
- `yield from` in `_iter_pdf_paths()` makes the helper a generator that streams paths in sorted order.
- `enumerate(reader.pages)` is used to get both the page object and the zero-based index needed for human-friendly 1-based page metadata.
- `if __name__ == "__main__":` makes the file runnable as a script without running the CLI when imported.

## Libraries
- LangChain is being used for `Document` objects and text splitting, but not for PDF parsing in the current ingestion code.
- `RecursiveCharacterTextSplitter` is the LangChain splitter used to break documents into overlapping chunks.
- `pypdf` provides direct PDF reading and text extraction through `PdfReader`.
- `pathlib.Path` is the standard library path type used for filesystem paths.
- `argparse` is used for the simple local CLI at the bottom of `app/ingest.py`.
- `langchain_core.documents.Document` stores the text and metadata that flow through the RAG pipeline.
- `langchain_text_splitters.RecursiveCharacterTextSplitter` preserves paragraph and line boundaries better than naive fixed-width splitting.
- `pypdf.PdfReader` is a lightweight choice for local PDF ingestion when OCR is not yet part of the pipeline.

## Conventions
- Use concise bullet points for durable project knowledge.
- Record exact repo paths and skill paths when they matter.
- Use `knowledge.md` as the canonical project memory file for this repo.

## Commands

## File Locations
- `app/ingest.py` contains the current ingestion helpers.
- `skills/knowledge-capture/SKILL.md` defines the capture workflow.
- `skills/knowledge-capture/agents/openai.yaml` contains the skill UI metadata.
- `.agents/` exists at the repo root but is not where this project’s skill metadata lives.

## Open Questions
