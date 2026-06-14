"""Create, populate, and persist the local Chroma vector index.

This module is the storage stage of the RAG ingestion pipeline. It accepts
chunk-level LangChain ``Document`` objects produced by ``app.ingest``, embeds
their text through a provider-neutral client from ``app.embeddings``, and
stores the resulting vectors and metadata in a persistent Chroma collection.
The saved collection can later be reopened by the retrieval layer for
similarity search.

Document identifiers are generated deterministically from source metadata,
chunk position, and content. Reprocessing an unchanged, identically ordered
corpus therefore produces the same identifiers, which gives indexing code a
stable reference for each chunk. Callers may also provide their own IDs when
upserting documents.

Persistence defaults to the repository's ``vectorstore`` directory and the
``documents`` collection. Opening a store creates the persistence directory
when necessary. Populating a store performs external embedding requests and
writes Chroma data to disk; this module does not load or split source files.
"""

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
    """Open or create a persistent Chroma vectorstore.

    The persistence directory is normalized to a ``Path`` and created before
    Chroma is initialized. The caller may provide an existing embedding client
    to keep indexing and retrieval on the same model. If no client is supplied,
    ``get_embeddings`` resolves one from environment configuration.

    Args:
        persist_dir: Directory where Chroma stores collection data. Relative
            paths are interpreted from the process working directory.
        collection_name: Chroma collection to open or create.
        embeddings: Optional embedding client used to encode documents and
            queries. When omitted, environment configuration selects one.

    Returns:
        A LangChain ``Chroma`` wrapper connected to the requested persistent
        collection.

    Raises:
        OSError: If the persistence directory cannot be created or accessed.
        RuntimeError: If no embedding provider is configured.
        Exception: Propagates initialization errors raised by Chroma or the
            embedding integration.

    Side effects:
        Creates the persistence directory and may initialize Chroma files on
        disk.
    """

    persist_path = Path(persist_dir)  # Normalize the persistence path.
    persist_path.mkdir(parents=True, exist_ok=True)  # Create the directory before opening the store.

    embedding_client = embeddings or get_embeddings()  # Use the caller's client or resolve one from env.

    return Chroma(  # Construct the LangChain Chroma wrapper.
        collection_name=collection_name,  # Keep the collection name explicit.
        embedding_function=embedding_client,  # Give Chroma the embedding implementation.
        persist_directory=str(persist_path),  # Store the index on disk for reuse.
    )


def _document_id(document: Document, position: int) -> str:
    """Create a deterministic identifier for one chunk document.

    The identifier hashes the document's source path, page number, chunk
    index, and complete text content. ``chunk_index`` metadata is preferred
    when present; otherwise the document's position in the current sequence is
    used. Including content ensures that edited chunks receive new IDs, while
    including metadata distinguishes equal text found in different locations.

    Args:
        document: Chunk-level document containing text and source metadata.
        position: Deterministic fallback index within the current sequence.

    Returns:
        A hexadecimal SHA-1 digest suitable for use as a Chroma document ID.

    Side effects:
        None.
    """

    source = str(document.metadata.get("source", ""))  # Capture the original PDF path when available.
    page = str(document.metadata.get("page", ""))  # Capture the source page number when available.
    chunk = str(document.metadata.get("chunk_index", position))  # Prefer an explicit chunk index if present.
    payload = "\n".join([source, page, chunk, document.page_content])  # Combine the stable fields into one string.
    return sha1(payload.encode("utf-8")).hexdigest()  # Hash the payload into a compact identifier.


def make_document_ids(documents: Sequence[Document]) -> list[str]:
    """Create deterministic Chroma IDs for an ordered document sequence.

    Each document is paired with its zero-based position and passed to
    ``_document_id``. Stable input ordering is important when documents do not
    contain explicit ``chunk_index`` metadata because their positions become
    part of the generated identifiers.

    Args:
        documents: Ordered chunk documents for which IDs are required.

    Returns:
        One deterministic identifier per document, preserving input order.

    Side effects:
        None.
    """

    return [_document_id(document, index) for index, document in enumerate(documents)]  # Build one ID per chunk.


def upsert_documents(
    vectorstore: Chroma,
    documents: Sequence[Document],
    ids: Sequence[str] | None = None,
) -> list[str]:
    """Embed and add chunk documents to an existing Chroma collection.

    The input sequence is materialized once so document counting, ID creation,
    and storage use the same stable order. Caller-provided IDs are accepted
    when they match the document count; otherwise deterministic IDs are
    generated from each document. Chroma embeds and stores the documents, then
    the collection is explicitly persisted to disk.

    Args:
        vectorstore: Initialized Chroma collection that will receive the
            documents.
        documents: Ordered chunk documents containing text and metadata.
        ids: Optional IDs corresponding positionally to ``documents``.

    Returns:
        The document IDs reported by Chroma after storage.

    Raises:
        ValueError: If the number of supplied IDs differs from the number of
            documents.
        Exception: Propagates embedding, Chroma storage, and persistence
            failures.

    Side effects:
        Sends document text to the configured embedding service and writes the
        resulting vectors, documents, metadata, and IDs to persistent storage.
    """

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
    """Initialize a persistent Chroma store and populate it with documents.

    This convenience function combines ``init_chroma`` and
    ``upsert_documents`` for the index-build workflow. It opens or creates the
    requested collection, embeds the supplied chunk documents, stores them with
    deterministic IDs, and returns both the usable store and those IDs.

    The function does not clear an existing collection. Rebuild behavior, such
    as deleting an old persistence directory, remains the responsibility of
    the calling script.

    Args:
        documents: Ordered chunk documents ready for embedding and storage.
        persist_dir: Directory where Chroma persists collection data.
        collection_name: Chroma collection to open or create.
        embeddings: Optional preconfigured embedding client. When omitted,
            environment configuration selects one.

    Returns:
        A tuple containing the initialized Chroma vectorstore and the list of
        stored document IDs.

    Raises:
        OSError: If the persistence directory cannot be created or written.
        RuntimeError: If no embedding provider is configured.
        Exception: Propagates embedding and Chroma storage failures.

    Side effects:
        Creates or opens persistent Chroma data, sends document text to the
        configured embedding service, and writes index updates to disk.
    """

    vectorstore = init_chroma(  # Open or create the persistent store first.
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    stored_ids = upsert_documents(vectorstore, documents)  # Write the documents into the store.
    return vectorstore, stored_ids  # Return the ready-to-query store and the new IDs.
