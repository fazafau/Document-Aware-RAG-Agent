"""Create and use embedding clients for the RAG pipeline.

This module provides one provider-neutral entry point for converting document
text and user queries into numerical vectors. It supports both Azure OpenAI and
the standard OpenAI API through LangChain's embedding integrations.

Configuration is read from environment variables after loading a local
``.env`` file when one is available. Callers may select a provider explicitly
with ``EMBEDDINGS_PROVIDER`` or allow the module to infer the provider from the
available credentials. Azure configuration takes precedence when both provider
configurations are present and no provider is selected explicitly.

The resulting embedding client is shared by the ingestion and retrieval sides
of the application. Document chunks and search queries must use compatible
embedding models so their vectors can be compared correctly in the
vectorstore. This module performs no persistence itself; it only creates
clients and delegates embedding requests to them.
"""

from __future__ import annotations  # Delay evaluation of type hints.

import os  # Read provider settings from environment variables.
from collections.abc import Sequence  # Describe text collections cleanly.
from typing import TypeAlias  # Declare a readable alias for the embedding client type.

from dotenv import load_dotenv  # Load local .env values when present.
from langchain_openai import AzureOpenAIEmbeddings  # Support Azure OpenAI embeddings.
from langchain_openai import OpenAIEmbeddings  # Support standard OpenAI embeddings.

EmbeddingClient: TypeAlias = OpenAIEmbeddings | AzureOpenAIEmbeddings  # Keep the return type readable.


def _load_environment() -> None:
    """Load configuration values from a local ``.env`` file.

    ``python-dotenv`` searches for an applicable file and adds its values to
    the process environment without requiring every caller to load it first.
    Existing environment variables retain their normal precedence according to
    ``load_dotenv`` defaults.

    Returns:
        None.

    Side effects:
        May add values from a local ``.env`` file to ``os.environ``.
    """

    load_dotenv()  # Pull local configuration into the current process.


def _is_azure_configured() -> bool:
    """Check whether the minimum Azure OpenAI settings are available.

    A usable Azure embedding client requires an endpoint, API key, and
    embedding deployment name. Optional values such as the model and API
    version are not part of this readiness check because defaults or provider
    behavior may supply them later.

    Returns:
        ``True`` when all required Azure environment variables contain
        non-empty values; otherwise ``False``.
    """

    return bool(  # Return True only when the Azure endpoint and deployment are present.
        os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    )


def _is_openai_configured() -> bool:
    """Check whether the minimum standard OpenAI settings are available.

    The standard OpenAI client only requires an API key at selection time.
    Model and endpoint settings remain optional because this module supplies a
    default model and supports the library's default API endpoint.

    Returns:
        ``True`` when ``OPENAI_API_KEY`` contains a non-empty value; otherwise
        ``False``.
    """

    return bool(os.getenv("OPENAI_API_KEY"))  # OpenAI only needs an API key to start.


def get_embeddings() -> EmbeddingClient:
    """Create an embedding client from the current environment configuration.

    The function first loads local environment values and normalizes
    ``EMBEDDINGS_PROVIDER``. An explicit value of ``azure`` or ``openai``
    selects that provider. When the setting is omitted, a complete Azure
    configuration is preferred, followed by a standard OpenAI configuration.

    Azure clients use ``AZURE_OPENAI_ENDPOINT``,
    ``AZURE_OPENAI_API_KEY``, and
    ``AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`` as required values. Standard OpenAI
    clients require ``OPENAI_API_KEY``. Both providers default to
    ``text-embedding-3-small`` unless their model environment variable
    overrides it.

    Returns:
        A configured LangChain ``AzureOpenAIEmbeddings`` or
        ``OpenAIEmbeddings`` client suitable for document and query
        embeddings.

    Raises:
        KeyError: If a provider is explicitly selected but one of the required
            environment variables accessed for that provider is missing.
        RuntimeError: If no supported provider can be selected from the
            environment.

    Side effects:
        Loads values from a local ``.env`` file into the process environment.
        No API request is made until the returned client is used.
    """

    _load_environment()  # Make local .env values available before reading config.

    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "").strip().lower()  # Allow explicit provider selection.

    if provider == "azure" or (not provider and _is_azure_configured()):  # Prefer Azure when configured.
        return AzureOpenAIEmbeddings(  # Build the Azure-backed embedding client.
            model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),  # Keep the model explicit.
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],  # Require a deployment name.
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # Let Azure use the configured API version.
            api_key=os.environ["AZURE_OPENAI_API_KEY"],  # Pass the Azure API key directly.
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # Point at the Azure endpoint.
            openai_api_type=os.getenv("AZURE_OPENAI_API_TYPE", "azure"),  # Preserve Azure-specific API type hints.
        )

    if provider == "openai" or (not provider and _is_openai_configured()):  # Fall back to standard OpenAI.
        return OpenAIEmbeddings(  # Build the standard OpenAI embedding client.
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),  # Use a modern default model.
            api_key=os.environ["OPENAI_API_KEY"],  # Require the API key.
            base_url=os.getenv("OPENAI_API_BASE"),  # Allow custom OpenAI-compatible endpoints.
            api_version=os.getenv("OPENAI_API_VERSION"),  # Preserve optional version pinning.
            openai_api_type=os.getenv("OPENAI_API_TYPE"),  # Preserve compatibility with alternate OpenAI targets.
        )

    raise RuntimeError(  # Fail clearly when no supported embedding configuration exists.
        "Configure EMBEDDINGS_PROVIDER=azure or set OPENAI_API_KEY / Azure OpenAI variables."
    )


def embed_texts(texts: Sequence[str], embeddings: EmbeddingClient | None = None) -> list[list[float]]:
    """Convert an ordered collection of document texts into vectors.

    This helper is intended for document ingestion and indexing. It preserves
    the input order so each returned vector corresponds to the text at the same
    position. A caller-supplied client can be reused across batches; otherwise
    the provider is resolved from the environment.

    Args:
        texts: Ordered text values to embed as documents.
        embeddings: Optional preconfigured embedding client. When omitted,
            ``get_embeddings`` creates one from environment settings.

    Returns:
        A list containing one floating-point embedding vector for each input
        text, in matching order.

    Raises:
        RuntimeError: If no embedding provider is configured.
        Exception: Propagates authentication, rate-limit, validation, and
            network errors raised by the selected embedding provider.

    Side effects:
        Sends the supplied text to the configured external embedding service.
    """

    client = embeddings or get_embeddings()  # Reuse the caller's client when provided.
    return client.embed_documents(list(texts))  # Delegate batching to the LangChain embedding client.


def embed_query(text: str, embeddings: EmbeddingClient | None = None) -> list[float]:
    """Convert one retrieval query into an embedding vector.

    Query embedding uses the provider's query-specific method so retrieval
    remains compatible with document vectors created by the same client and
    model. A supplied client avoids rebuilding configuration for repeated
    searches; otherwise the environment determines the provider.

    Args:
        text: Query text to convert into a vector.
        embeddings: Optional preconfigured embedding client. When omitted,
            ``get_embeddings`` creates one from environment settings.

    Returns:
        The floating-point embedding vector produced for the query.

    Raises:
        RuntimeError: If no embedding provider is configured.
        Exception: Propagates authentication, rate-limit, validation, and
            network errors raised by the selected embedding provider.

    Side effects:
        Sends the query text to the configured external embedding service.
    """

    client = embeddings or get_embeddings()  # Reuse the caller's client when provided.
    return client.embed_query(text)  # Use the query-specific embedding method.
