"""Build and persist the local Chroma index from the repository PDF corpus.

This script implements Step 4 of ``RAG_AGENT_PLAN.md`` and connects the earlier
pipeline stages into one runnable command:

1. Load page-level PDF documents from the configured source directory.
2. Clean and split those pages into deterministic, overlapping text chunks.
3. Create the configured Azure OpenAI or standard OpenAI embedding client.
4. Open a persistent Chroma collection.
5. Embed and upsert the chunks into that collection.

The command defaults to reading ``data/docs`` and writing the index under
``vectorstore``. ``--persist-dir`` can select another output directory, while
``--force`` removes an existing output directory before indexing so the build
starts from an empty Chroma store. Destructive removal is restricted to a
specific directory below the repository root; the repository root and source
document directory cannot be selected as force-removal targets.

Embedding provider configuration is loaded by ``app.embeddings`` from process
environment variables and an optional local ``.env`` file. Building the index
therefore performs external API requests and may incur provider usage costs.
The resulting files are consumed by the retrieval and QA stages of the RAG
application.

Run the script from the repository root:

    python scripts/build_index.py
    python scripts/build_index.py --force --persist-dir vectorstore
"""

from __future__ import annotations  # Delay evaluation of type hints.

import argparse  # Parse command-line build options.
import shutil  # Remove an existing index during an explicit forced rebuild.
import sys  # Make the repository package importable when run as a file.
from pathlib import Path  # Normalize and validate source and persistence paths.


# Resolve defaults relative to the repository instead of the caller's directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "docs"  # Use the repository PDF corpus.
DEFAULT_PERSIST_DIR = REPO_ROOT / "vectorstore"  # Keep the local index in the planned location.
DEFAULT_PATTERN = "*.pdf"  # Read top-level PDFs by default.
DEFAULT_CHUNK_SIZE = 1000  # Balance retrieval context with embedding precision.
DEFAULT_CHUNK_OVERLAP = 200  # Preserve context around neighboring chunk boundaries.
DEFAULT_COLLECTION_NAME = "documents"  # Keep one predictable initial Chroma collection.

if str(REPO_ROOT) not in sys.path:  # Support direct execution from any working directory.
    sys.path.insert(0, str(REPO_ROOT))  # Put local application modules before installed packages.

from app.embeddings import get_embeddings  # Resolve the configured embedding provider.
from app.ingest import load_documents  # Load and clean PDF pages.
from app.ingest import split_documents  # Split page documents into embedding-ready chunks.
from app.vectorstore import init_chroma  # Open the persistent Chroma collection.
from app.vectorstore import upsert_documents  # Embed and store chunk documents.


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for local index builds.

    The parser exposes corpus selection, chunking, collection, and persistence
    settings so the same script can support normal builds and clean rebuilds.
    Path values remain strings during parsing and are normalized by
    ``build_index`` before filesystem operations begin.

    Returns:
        A configured ``ArgumentParser`` ready to parse command-line arguments.

    Side effects:
        None.
    """

    parser = argparse.ArgumentParser(  # Create the command's top-level help text.
        description="Load PDFs, create embeddings, and persist a Chroma index."
    )
    parser.add_argument(  # Let callers select a different PDF corpus.
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Directory containing PDF files (default: {DEFAULT_SOURCE_DIR}).",
    )
    parser.add_argument(  # Control which source files are ingested.
        "--pattern",
        default=DEFAULT_PATTERN,
        help='Glob pattern for source files, such as "*.pdf" or "**/*.pdf".',
    )
    parser.add_argument(  # Allow chunk-size tuning without code changes.
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target chunk size in characters (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(  # Allow overlap tuning for retrieval context.
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Characters shared by neighboring chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(  # Fulfill the plan's configurable persistence requirement.
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help=f"Directory for the Chroma index (default: {DEFAULT_PERSIST_DIR}).",
    )
    parser.add_argument(  # Support alternate Chroma collections when needed.
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(  # Fulfill the plan's clean-rebuild requirement.
        "--force",
        action="store_true",
        help="Delete the existing persistence directory before building.",
    )
    return parser  # Return the complete CLI definition.


def _validate_persist_dir(persist_dir: Path, source_dir: Path) -> None:
    """Validate that a persistence path is safe for normal and forced builds.

    The index must live inside the repository so an accidental CLI value cannot
    direct recursive deletion elsewhere. The path must also be more specific
    than the repository root and must not equal or contain the source document
    directory. These checks make ``--force`` safe while preserving the intended
    default under ``vectorstore``.

    Args:
        persist_dir: Resolved directory where Chroma data will be stored.
        source_dir: Resolved directory containing source documents.

    Raises:
        ValueError: If the persistence path is outside the repository, equals
            the repository root, or overlaps the source document directory.

    Side effects:
        None.
    """

    if persist_dir == REPO_ROOT:  # Never allow the project itself to become an index target.
        raise ValueError("persist_dir must not be the repository root")

    if not persist_dir.is_relative_to(REPO_ROOT):  # Keep recursive removal within this project.
        raise ValueError(f"persist_dir must be inside the repository: {REPO_ROOT}")

    paths_overlap = (  # Detect either direction of source and output nesting.
        persist_dir == source_dir
        or persist_dir.is_relative_to(source_dir)
        or source_dir.is_relative_to(persist_dir)
    )
    if paths_overlap:  # Protect PDFs from index writes and forced deletion.
        raise ValueError("persist_dir must not overlap source_dir")


def build_index(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    pattern: str = DEFAULT_PATTERN,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force: bool = False,
) -> tuple[int, int]:
    """Build a persistent Chroma index from PDF documents.

    The function executes the Step 4 pipeline explicitly: it loads page
    documents, splits them into chunks, initializes the configured embedding
    client, opens Chroma, and upserts every chunk. Paths are resolved before
    use, and the persistence target is validated before an optional forced
    rebuild removes existing data.

    Args:
        source_dir: Directory containing source PDF files.
        pattern: Glob pattern used to select source documents.
        chunk_size: Target number of characters in each text chunk.
        chunk_overlap: Number of characters shared by neighboring chunks.
        persist_dir: Directory where the Chroma collection will be stored.
        collection_name: Name of the Chroma collection to populate.
        force: Remove the existing persistence directory before indexing.

    Returns:
        A tuple containing the number of loaded page documents and the number
        of chunk IDs stored by Chroma.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        NotADirectoryError: If the source path exists but is not a directory.
        ValueError: If arguments are empty, chunk settings are invalid, no
            matching text can be loaded, or the persistence path is unsafe.
        RuntimeError: If no embedding provider is configured.
        OSError: If the persistence directory cannot be removed or written.
        Exception: Propagates PDF parsing, embedding provider, and Chroma
            storage errors.

    Side effects:
        May delete an existing persistence directory when ``force`` is true,
        sends document text to the configured embedding service, and writes a
        persistent Chroma index to disk.
    """

    source_path = Path(source_dir).expanduser().resolve()  # Normalize the corpus location.
    persist_path = Path(persist_dir).expanduser().resolve()  # Normalize the index location.

    if not source_path.exists():  # Report missing input before provider initialization.
        raise FileNotFoundError(f"Document directory does not exist: {source_path}")
    if not source_path.is_dir():  # Require a directory because ingestion uses glob matching.
        raise NotADirectoryError(f"Document source is not a directory: {source_path}")
    if not pattern.strip():  # Prevent an ambiguous empty file-selection pattern.
        raise ValueError("pattern must not be empty")
    if not collection_name.strip():  # Require a usable Chroma collection name.
        raise ValueError("collection_name must not be empty")

    _validate_persist_dir(persist_path, source_path)  # Validate before any destructive operation.

    if force and persist_path.exists():  # Start from an empty index only when requested.
        shutil.rmtree(persist_path)  # Remove the complete prior Chroma persistence directory.

    page_documents = load_documents(  # Load and clean source PDFs into page documents.
        source_dir=source_path,
        pattern=pattern,
    )
    chunk_documents = split_documents(  # Produce retrieval-sized chunks with preserved metadata.
        documents=page_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = get_embeddings()  # Create the provider-specific embedding client.
    vectorstore = init_chroma(  # Open or initialize the requested persistent collection.
        persist_dir=persist_path,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    stored_ids = upsert_documents(  # Embed and persist every chunk in stable order.
        vectorstore=vectorstore,
        documents=chunk_documents,
    )

    return len(page_documents), len(stored_ids)  # Return concise build statistics to the caller.


def main() -> None:
    """Parse CLI arguments, build the index, and print a concise summary.

    Command-line failures are intentionally allowed to surface as exceptions so
    local users and automation receive a non-zero exit status with the original
    diagnostic. A successful run reports the page count, stored chunk count,
    persistence directory, and collection name.

    Returns:
        None.

    Side effects:
        Parses process arguments, performs the complete index-build workflow,
        writes status information to standard output, and may modify the
        configured persistence directory.
    """

    args = _build_arg_parser().parse_args()  # Parse the requested build configuration.
    page_count, stored_count = build_index(  # Run the complete indexing pipeline.
        source_dir=args.source_dir,
        pattern=args.pattern,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        force=args.force,
    )
    persist_path = Path(args.persist_dir).expanduser().resolve()  # Match the normalized build path.

    print(f"Loaded page documents: {page_count}")  # Report successfully parsed PDF pages.
    print(f"Stored chunk documents: {stored_count}")  # Report vectors persisted to Chroma.
    print(f"Persisted vectorstore: {persist_path}")  # Show where retrieval should load the index.
    print(f"Collection name: {args.collection_name}")  # Show which Chroma collection was populated.


if __name__ == "__main__":  # Run the CLI only when this file is executed directly.
    main()  # Start the index build.
