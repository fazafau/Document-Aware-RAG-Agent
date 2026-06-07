"""Build the local document index from PDFs and persist it to Chroma."""

from __future__ import annotations  # Delay evaluation of type hints.

import argparse  # Parse command-line options.
import shutil  # Remove an existing vectorstore when rebuilding from scratch.
import sys  # Add the repository root to the import path when run as a script.
from pathlib import Path  # Work with repository paths explicitly.

REPO_ROOT = Path(__file__).resolve().parents[1]  # Find the project root from the script location.
if str(REPO_ROOT) not in sys.path:  # Make the app package importable when run directly.
    sys.path.insert(0, str(REPO_ROOT))  # Put the repo root at the front of sys.path.

from app.embeddings import get_embeddings  # Resolve the configured embedding provider.
from app.ingest import ingest_documents  # Load and split PDFs into chunk documents.
from app.vectorstore import build_vectorstore  # Create and populate the Chroma store.


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI for the index creation script."""

    parser = argparse.ArgumentParser(description="Build the local Chroma index from PDFs.")  # Name the script clearly.
    parser.add_argument("--source-dir", default="data/docs", help="Directory containing PDF files.")  # Choose the source folder.
    parser.add_argument("--pattern", default="*.pdf", help='Glob pattern for documents, for example "*.pdf".')  # Filter input files.
    parser.add_argument("--chunk-size", type=int, default=1000, help="Target chunk size in characters.")  # Tune chunk size.
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Characters of overlap between neighboring chunks.")  # Tune overlap.
    parser.add_argument("--persist-dir", default="vectorstore", help="Directory where the Chroma index will be stored.")  # Set the output directory.
    parser.add_argument("--collection-name", default="documents", help="Name of the Chroma collection to populate.")  # Set the collection name.
    parser.add_argument("--force", action="store_true", help="Delete the existing vectorstore before rebuilding.")  # Allow a clean rebuild.
    return parser  # Hand the parser back to the caller.


def main() -> None:
    """Build a fresh persistent vectorstore from the current document set."""

    args = _build_arg_parser().parse_args()  # Read the requested build settings.
    persist_dir = Path(args.persist_dir)  # Normalize the output directory.

    if args.force and persist_dir.exists():  # Rebuild from scratch when the user asks for it.
        shutil.rmtree(persist_dir)  # Remove the prior persisted Chroma data.

    embeddings = get_embeddings()  # Resolve the embedding provider from the environment.
    page_documents = ingest_documents(  # Load and split the current PDF corpus.
        source_dir=args.source_dir,
        pattern=args.pattern,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    vectorstore, stored_ids = build_vectorstore(  # Create and populate the persistent store.
        documents=page_documents,
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        embeddings=embeddings,
    )

    print(f"Loaded chunk documents: {len(page_documents)}")  # Report the number of chunk documents processed.
    print(f"Persisted vectorstore: {persist_dir}")  # Report the output directory.
    print(f"Collection name: {args.collection_name}")  # Report the collection name for clarity.
    print(f"Stored IDs: {len(stored_ids)}")  # Confirm how many chunks were stored.
    print(f"Vectorstore type: {vectorstore.__class__.__name__}")  # Report the store implementation.


if __name__ == "__main__":  # Run the build only when the script is executed directly.
    main()  # Start the index build.
