"""Vectorstore helpers for persisting and updating the local Chroma index."""

from __future__ import annotations  # Delay evaluation of type hints.

from collections.abc import Sequence  # Accept any ordered collection of documents.
from hashlib import sha1  # Build deterministic document identifiers.
from pathlib import Path  # Work with persistence directories safely.

from langchain_core.documents import Document  # Keep the vectorstore input type explicit.
from langchain_community.vectorstores import Chroma  # Use Chroma as the local persistent store.

from app.embeddings import EmbeddingClient  # Reuse the provider-agnostic embedding type.
from app.embeddings import get_embeddings  # Build an embedding client when one is not supplied.

DEFAULT_PERSIST_DIR = Path("vectorstore")  # Keep the persistent index in a predictable location.
DEFAULT_COLLECTION_NAME = "documents"  # Use a single collection name for the initial pipeline.


def init_chroma(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: EmbeddingClient | None = None,
) -> Chroma:
    """Create a persistent Chroma vectorstore instance."""

    persist_path = Path(persist_dir)  # Normalize the persistence path.
    persist_path.mkdir(parents=True, exist_ok=True)  # Create the directory before opening the store.

    embedding_client = embeddings or get_embeddings()  # Use the caller's client or resolve one from env.

    return Chroma(  # Construct the LangChain Chroma wrapper.
        collection_name=collection_name,  # Keep the collection name explicit.
        embedding_function=embedding_client,  # Give Chroma the embedding implementation.
        persist_directory=str(persist_path),  # Store the index on disk for reuse.
    )


def _document_id(document: Document, position: int) -> str:
    """Create a stable identifier for one chunk document."""

    source = str(document.metadata.get("source", ""))  # Capture the original PDF path when available.
    page = str(document.metadata.get("page", ""))  # Capture the source page number when available.
    chunk = str(document.metadata.get("chunk_index", position))  # Prefer an explicit chunk index if present.
    payload = "\n".join([source, page, chunk, document.page_content])  # Combine the stable fields into one string.
    return sha1(payload.encode("utf-8")).hexdigest()  # Hash the payload into a compact identifier.


def make_document_ids(documents: Sequence[Document]) -> list[str]:
    """Create deterministic IDs for a sequence of chunk documents."""

    return [_document_id(document, index) for index, document in enumerate(documents)]  # Build one ID per chunk.


def upsert_documents(
    vectorstore: Chroma,
    documents: Sequence[Document],
    ids: Sequence[str] | None = None,
) -> list[str]:
    """Add chunk documents to an existing Chroma vectorstore."""

    chunk_documents = list(documents)  # Materialize the documents so we can count and index them consistently.
    document_ids = list(ids) if ids is not None else make_document_ids(chunk_documents)  # Use caller IDs or generate them.

    if len(document_ids) != len(chunk_documents):  # Guard against mismatched inputs.
        raise ValueError("ids must match the number of documents")  # Fail loudly when the caller data is inconsistent.

    stored_ids = vectorstore.add_documents(chunk_documents, ids=document_ids)  # Delegate storage to Chroma.
    vectorstore.persist()  # Flush the updated collection to disk.
    return stored_ids  # Hand the inserted IDs back to the caller.


def build_vectorstore(
    documents: Sequence[Document],
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: EmbeddingClient | None = None,
) -> tuple[Chroma, list[str]]:
    """Create a Chroma store and populate it with documents."""

    vectorstore = init_chroma(  # Open or create the persistent store first.
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    stored_ids = upsert_documents(vectorstore, documents)  # Write the documents into the store.
    return vectorstore, stored_ids  # Return the ready-to-query store and the new IDs.
