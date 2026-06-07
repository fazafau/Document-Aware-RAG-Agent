"""Document ingestion helpers for the local RAG pipeline.

This module is the first stage of the pipeline described in RAG_AGENT_PLAN.md:

1. Read source PDFs from disk.
2. Convert each PDF page into a LangChain ``Document``.
3. Split those page documents into smaller overlapping chunks.

The output of ``split_documents`` is intentionally ready for the next pipeline
step: embeddings and vectorstore upsert.
"""

from __future__ import annotations  # Delay type-hint evaluation until runtime.

import argparse  # Build the small command-line entry point.
from pathlib import Path  # Represent filesystem paths explicitly.
from typing import Iterable  # Describe generator-like return values.

from langchain_core.documents import Document  # Store text plus metadata.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split text safely.
from pypdf import PdfReader  # Read PDF files and extract page text.


# Keep the default corpus location in one place for easy reuse.
DEFAULT_DOCS_DIR = Path("data/docs")


def _clean_page_text(text: str) -> str:
    """Normalize text extracted from a PDF page.

    PDF extraction often gives us uneven whitespace. This small cleanup keeps
    the original word order, but collapses repeated spaces and blank lines so
    the splitter receives cleaner paragraphs.
    """

    lines = []  # Collect cleaned non-empty lines.
    for line in text.splitlines():
        stripped = " ".join(line.split())  # Collapse repeated internal whitespace.
        if stripped:
            lines.append(stripped)  # Keep only meaningful content.

    return "\n".join(lines)  # Rebuild the page text with clean line breaks.


def _iter_pdf_paths(source_dir: Path, pattern: str) -> Iterable[Path]:
    """Yield PDF paths in a stable order.

    Sorting matters because ingestion should be repeatable: if two developers
    run the same index build, they should see documents processed in the same
    order.
    """

    yield from sorted(source_dir.glob(pattern))  # Yield matches in a stable order.


def load_documents(
    source_dir: str | Path = DEFAULT_DOCS_DIR,
    pattern: str = "*.pdf",
) -> list[Document]:
    """Load PDFs from ``source_dir`` into page-level LangChain documents.

    Args:
        source_dir: Directory containing PDF files.
        pattern: Glob pattern used to select files. The default reads only PDF
            files in the top-level directory. Use ``"**/*.pdf"`` for recursive
            loading later if you organize documents into subfolders.

    Returns:
        A list of ``Document`` objects. Each object represents one PDF page and
        includes metadata that will survive into vectorstore chunks:
        ``source`` path, ``file_name``, ``page`` number, and ``total_pages``.

    Raises:
        FileNotFoundError: If ``source_dir`` does not exist.
        ValueError: If no matching PDFs are found or no text can be extracted.
    """

    docs_path = Path(source_dir)  # Normalize the source directory to Path.
    if not docs_path.exists():
        raise FileNotFoundError(f"Document directory does not exist: {docs_path}")  # Fail fast.

    pdf_paths = list(_iter_pdf_paths(docs_path, pattern))  # Materialize the matching files.
    if not pdf_paths:
        raise ValueError(f"No PDF files matched {pattern!r} in {docs_path}")  # Require input data.

    documents: list[Document] = []  # Accumulate page-level documents here.

    for pdf_path in pdf_paths:
        # Parse one PDF at a time to keep memory use and control flow simple.
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)  # Record the total page count for metadata.

        for page_index, page in enumerate(reader.pages):
            # Extract text; scanned pages may return None.
            raw_text = page.extract_text() or ""
            cleaned_text = _clean_page_text(raw_text)  # Normalize the extracted text.

            if not cleaned_text:
                continue  # Skip empty pages instead of creating blank documents.

            documents.append(
                Document(
                    page_content=cleaned_text,  # Store the cleaned page text.
                    metadata={
                        "source": str(pdf_path),  # Preserve the original file path.
                        "file_name": pdf_path.name,  # Keep the short file name too.
                        "page": page_index + 1,  # Use human-friendly 1-based pages.
                        "total_pages": total_pages,  # Keep the total page count.
                    },
                )
            )  # Save one LangChain document per extracted page.

    if not documents:
        raise ValueError(
            f"Found {len(pdf_paths)} PDF file(s), but no extractable text was loaded."
        )  # Tell the caller the corpus could not be parsed.

    return documents  # Return the page-level documents for downstream splitting.


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split page-level documents into overlapping text chunks.

    The vectorstore works best with chunks that are large enough to contain
    context but small enough to embed precisely. Overlap keeps important
    sentences near chunk boundaries from being separated too harshly.

    Args:
        documents: Page-level documents from ``load_documents``.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of characters repeated between neighboring
            chunks. This must be smaller than ``chunk_size``.

    Returns:
        Chunk-level ``Document`` objects. The original metadata is preserved,
        and LangChain adds chunk-specific metadata when available.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")  # Reject invalid chunk sizes.
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")  # Reject negative overlap.
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")  # Keep overlap bounded.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Target chunk size in characters.
        chunk_overlap=chunk_overlap,  # Preserve some shared context between chunks.
        separators=["\n\n", "\n", " ", ""],  # Prefer natural boundaries before raw characters.
    )  # Build the LangChain splitter with the chosen policy.

    return splitter.split_documents(documents)  # Let LangChain produce chunk documents.


def ingest_documents(
    source_dir: str | Path = DEFAULT_DOCS_DIR,
    pattern: str = "*.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Convenience wrapper that loads PDFs and returns chunked documents."""

    page_documents = load_documents(source_dir=source_dir, pattern=pattern)  # Load page docs first.
    return split_documents(
        page_documents,  # Feed the pages into the splitter.
        chunk_size=chunk_size,  # Reuse the requested chunk size.
        chunk_overlap=chunk_overlap,  # Reuse the requested overlap.
    )  # Return ready-to-embed chunks.


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the small CLI used for local ingestion smoke tests."""

    parser = argparse.ArgumentParser(
        description="Load PDFs and split them into RAG-ready chunks."  # Describe the CLI purpose.
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_DOCS_DIR),  # Default to the repo's docs directory.
        help="Directory containing PDF files.",  # Explain the source directory flag.
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",  # Read PDFs in the top-level folder by default.
        help='Glob pattern for documents, for example "*.pdf" or "**/*.pdf".',  # Document the pattern.
    )
    parser.add_argument(
        "--chunk-size",
        type=int,  # Parse the value as an integer.
        default=1000,  # Use a moderate default chunk size.
        help="Target chunk size in characters.",  # Describe the setting.
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,  # Parse the value as an integer.
        default=200,  # Keep a little context between chunks.
        help="Characters of overlap between neighboring chunks.",  # Describe the setting.
    )
    return parser  # Hand the configured parser back to the caller.


def main() -> None:
    """Run a quick ingestion preview from the command line.

    This does not write embeddings or a vectorstore yet; it only proves that PDF
    loading and chunking are working.
    """

    args = _build_arg_parser().parse_args()  # Read CLI arguments from the shell.
    page_documents = load_documents(source_dir=args.source_dir, pattern=args.pattern)  # Load pages.
    chunks = split_documents(
        page_documents,  # Pass the loaded pages into chunking.
        chunk_size=args.chunk_size,  # Use the user-supplied chunk size.
        chunk_overlap=args.chunk_overlap,  # Use the user-supplied overlap.
    )  # Produce chunked documents for inspection.

    print(f"Loaded page documents: {len(page_documents)}")  # Report how many pages were loaded.
    print(f"Created chunks: {len(chunks)}")  # Report how many chunks were produced.

    if chunks:
        first = chunks[0]  # Inspect the first chunk as a quick sanity check.
        print("\nFirst chunk metadata:")
        for key, value in first.metadata.items():
            print(f"  {key}: {value}")  # Show the preserved metadata.
        print("\nFirst chunk preview:")
        print(first.page_content[:500])  # Show a short text preview.


if __name__ == "__main__":  # Run the CLI only when executed directly.
    main()  # Start the ingestion smoke test.
