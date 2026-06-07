# Azure RAG Agent Plan

Plan from May 27, 2026.

## TL;DR

Build a local RAG pipeline first: ingest PDFs, create embeddings, persist a Chroma index, and run Retriever + QA. Then add an agent layer with a small toolset: web search, safe Python executor, filesystem/doc access, and diagram/flowchart generator.

Make the code provider-agnostic for OpenAI vs Azure OpenAI, containerize it, and prepare Azure deployment artifacts.

Key repo anchors: `pyproject.toml`, `main.py`, `docs`, `app`, `vectorstore`.

## Steps

### 1. Project Setup

Create `.env.example` documenting required environment variables:

```env
OPENAI_API_KEY=
OPENAI_API_BASE=
OPENAI_API_TYPE=
OPENAI_API_VERSION=
```

Add a `scripts/` directory for runnable examples.

### 2. Document Ingestion

Add `app/ingest.py` with `load_documents()` and `split_documents()` that:

- Reads PDFs from `docs`
- Uses `pypdf` or an `UnstructuredPDFLoader` style loader
- Splits text into chunks with overlap via a LangChain text splitter

Output: chunked documents ready for embedding.

### 3. Embeddings + Vectorstore Adapter

Add `app/embeddings.py` with a `get_embeddings(client_config)` abstraction exposing `embed_texts(texts)` using a provider-agnostic wrapper:

- `OpenAIEmbeddings`
- `AzureOpenAIEmbeddings`

Selection should be based on environment variables.

Add `app/vectorstore.py` with:

- `init_chroma(persist_dir)`
- `upsert_documents(chroma_client, docs)`

Use Chroma for local development and persist the index under `vectorstore`.

### 4. Build-Index Script

Add `scripts/build_index.py` that calls:

```text
load_documents() -> embed -> init_chroma() -> upsert
```

It should write the index to `vectorstore`.

Make it CLI-friendly with:

- `--force`
- `--persist-dir`

### 5. Retriever + QA Pipeline

Add `app/rag.py` with:

- `make_retriever(vectorstore_dir)`
- `qa_query(query, k=4)`

Use LangChain `RetrievalQA` or `RetrievalQAWithSources`, hooking into the provider-agnostic LLM wrapper.

Expose an `answer_query()` API-level function for reuse.

### 6. Minimal API / Local Server

Create `app/server.py` using FastAPI and expose:

- `POST /query` -> calls `qa_query`
- `POST /reindex` -> triggers `scripts/build_index.py` asynchronously or in the background

Update `main.py` to run:

```bash
uvicorn app.server:app
```

Alternatively, add `scripts/run_server.py`.

### 7. Agent Layer & Tools

Add `app/agent.py` implementing an agent orchestrator using LangChain agents or a custom loop with:

- Tool 1: `web_search_tool` - encapsulate web search, using a SERP API or fallback. Keep as optional.
- Tool 2: `python_executor` - sandboxed executor with time and memory limits; returns stdout and structured results.
- Tool 3: `local_doc_tool` - search/read documents via the retriever/QA.
- Tool 4: `flowchart_tool` - produce flowchart/diagram suggestions, initially as DOT or Mermaid text; rendering optional.

Design APIs:

```python
Agent.run(prompt, tools=...)
```

Keep tool interfaces small:

```python
run(input) -> result
```

### 8. Security & Sandboxing

For `python_executor`:

- Run a subprocess in a temporary directory
- Restrict filesystem access
- Limit CPU/time
- Sanitize outputs

Document risks in `README.md`.

Restrict any tool that allows network calls unless explicitly enabled by environment variable.

### 9. Tests and Examples

Add basic unit/integration tests for:

- Ingestion
- Embedding wrapper, mocked
- Vectorstore initialization
- `qa_query`

Add an example notebook or `scripts/demo_query.py`.

### 10. Containerization & Infra Prep

Add a `Dockerfile` for the app:

- Use an official Python 3.12 image
- Copy app files
- Mount `vectorstore` persist volume at runtime

Add Azure deployment notes:

- Local-first development
- Container + ACR -> ACI or Web App for Containers

Add `azure-deploy/` with ARM/Bicep templates or README deployment steps.

### 11. Documentation

Update `README.md` with:

- Setup
- `.env` usage
- Building the index
- Running the server
- Running agent tools
- Cloud deployment checklist

Add troubleshooting notes for:

- Native dependencies such as Chroma
- Python version, `>=3.12`

## Verification

Build the index and run a local query.

If preferred, add a concrete `requirements.txt` and exact install commands.

## Decisions

- Vectorstore: Use Chroma for local development, while keeping an abstraction to swap to Azure Cognitive Search or a managed database for production.
- Provider: Design a provider-agnostic LLM wrapper so switching between OpenAI and Azure OpenAI is an environment configuration change.
- Deployment: Target local-first development with containerization. Preferred Azure target is container-based deployment: ACR + ACI or AKS later.

## Critical Files To Add Or Change

- `.env.example`
- `scripts/build_index.py`
- `scripts/demo_query.py`
- `app/ingest.py`
- `app/embeddings.py`
- `app/vectorstore.py`
- `app/rag.py`
- `app/server.py`
- `app/agent.py`
- `Dockerfile`
- `azure-deploy/`
- `README.md`

## Next Options

1. Produce the initial set of files for the RAG pipeline: ingest, embeddings, vectorstore, build script, and demo.
2. Produce the agent scaffolding: tools and orchestrator.
3. Scaffold both the RAG pipeline and agent layer in one pass.

Recommended first step: implement the ingestion -> index -> QA pieces first.
