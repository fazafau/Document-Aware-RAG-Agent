"""Embedding helpers for the local and Azure OpenAI RAG pipeline."""

from __future__ import annotations  # Delay evaluation of type hints.

import os  # Read provider settings from environment variables.
from collections.abc import Sequence  # Describe text collections cleanly.
from typing import TypeAlias  # Declare a readable alias for the embedding client type.

from dotenv import load_dotenv  # Load local .env values when present.
from langchain_openai import AzureOpenAIEmbeddings  # Support Azure OpenAI embeddings.
from langchain_openai import OpenAIEmbeddings  # Support standard OpenAI embeddings.

EmbeddingClient: TypeAlias = OpenAIEmbeddings | AzureOpenAIEmbeddings  # Keep the return type readable.


def _load_environment() -> None:
    """Load environment variables from a local .env file if one exists."""

    load_dotenv()  # Pull local configuration into the current process.


def _is_azure_configured() -> bool:
    """Check whether the Azure embedding settings are available."""

    return bool(  # Return True only when the Azure endpoint and deployment are present.
        os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    )


def _is_openai_configured() -> bool:
    """Check whether the standard OpenAI embedding settings are available."""

    return bool(os.getenv("OPENAI_API_KEY"))  # OpenAI only needs an API key to start.


def get_embeddings() -> EmbeddingClient:
    """Create the embedding client based on the current environment."""

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
    """Embed a batch of texts with the configured embedding client."""

    client = embeddings or get_embeddings()  # Reuse the caller's client when provided.
    return client.embed_documents(list(texts))  # Delegate batching to the LangChain embedding client.


def embed_query(text: str, embeddings: EmbeddingClient | None = None) -> list[float]:
    """Embed a single query string with the configured embedding client."""

    client = embeddings or get_embeddings()  # Reuse the caller's client when provided.
    return client.embed_query(text)  # Use the query-specific embedding method.
