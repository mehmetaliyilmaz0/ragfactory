# ragfactory

Generate standalone Python RAG projects from YAML or JSON. Choose LangChain or
LlamaIndex, validate compatibility, and export editable code through a CLI or
REST API. Generated projects call upstream SDKs directly, with no `ragfactory`
runtime dependency.

## Quick start

Requires Python 3.11 or newer. Generated SDK integration tests use Python 3.12.

```bash
# From this checkout, install the refreshed generator.
pip install -e ".[api]"
ragfactory init --name my-rag --vector-db qdrant --retrieval dense --output ./my-rag
cd my-rag
pip install .
cp .env.example .env
# Set API keys in .env. Start Qdrant locally or set QDRANT_URL.
python ingestion.py ./documents
uvicorn api:app --host 127.0.0.1 --port 8000
```

For an embedded database use `--vector-db chromadb`. It needs no database server;
API-backed embeddings and generation still need credentials. Generated entrypoints
load `.env`; existing process environment variables take precedence.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What do these documents describe?","return_contexts":true}'
```

## Configuration

```yaml
name: support-rag
framework: langchain  # or llamaindex
indexing:
  chunking: {type: recursive, chunk_size: 512, chunk_overlap: 50}
  embedding: {type: openai, model: text-embedding-3-small}
  vector_db: {type: qdrant, collection_name: support-docs, url: 'http://localhost:6333'}
retrieval: {type: dense, top_k: 10}
generation:
  llm: {type: openai, model: gpt-5.4-mini, max_tokens: 2048}
```

```bash
ragfactory validate --config pipeline.yaml
ragfactory generate --config pipeline.yaml --output ./support-rag
```

LLM model IDs accept nonempty strings so new releases do not require a schema
change. Availability depends on provider, account, region, and SDK support.
Embedding dimensions must be known or explicitly supplied where supported.
Changing the embedding model or dimensions requires a new collection and full
reindexing. Temperature is omitted by default for providers with reasoning models
that restrict it.

## Commands and API

| Command | Purpose |
| --- | --- |
| `ragfactory init --name NAME` | Generate from defaults and optional component flags |
| `ragfactory init --name NAME --save-config pipeline.yaml` | Save configuration only |
| `ragfactory validate --config pipeline.yaml` | Check schema and compatibility |
| `ragfactory generate --config pipeline.yaml --output DIR` | Render and syntax-check output |
| `ragfactory options --json` | List component choices |
| `ragfactory api --host 127.0.0.1 --port 8000` | Serve factory API, requires `api` extra |

Factory endpoints: `POST /api/v1/validate`, `POST /api/v1/generate` returning ZIP,
and `GET /api/v1/schema`. The generated project's separate API exposes `POST /query`.

Output includes `pipeline.py`, `ingestion.py`, `api.py`, `config.yaml`,
`pyproject.toml`, `.env.example`, `Dockerfile`, and `README.md`. External database
choices add Compose; evaluation adds `eval.py`. Generating into an existing
directory can overwrite files. Back up edited output first.

## Compatibility and limits

The dependency matrix targets LangChain 1.x, LangGraph 1.x, and LlamaIndex core
0.14 with independently versioned integrations. Legacy LangChain retrievers use
`langchain-classic`. Retired Google embedding IDs `text-embedding-004` and
`embedding-001` are rejected with migration guidance.

- Dense retrieval is the baseline. LangChain hybrid uses a corpus sidecar saved
  during ingestion; keep it with the generated project. LlamaIndex native hybrid
  is restricted to the implemented database path.
- LangChain ingestion preserves explicit file globs and loads UTF-8 text formats,
  including `.txt`, `.md`, `.rst`, `.csv`, `.json`, `.jsonl`, `.yaml`, `.yml`,
  `.html`, `.htm`, `.xml`, and `.log`. Unsupported formats such as PDF and DOCX
  fail before indexing; add a parser to the generated script to support them.
  With no configured source, directory ingestion defaults to `**/*.txt`.
- Contextual, proposition, and late chunking remain parseable for migration but
  generation rejects their incomplete ingestion implementations. `--force`
  cannot bypass that check. Enabled CRAG and FLARE are also rejected.
- Agentic mode generates a graph/workflow scaffold; it does not implement
  autonomous planning, reflection, web search, or arbitrary tool execution.
- Parent-document and sentence-window retrieval require corresponding ingestion
  metadata/docstores. Routing, compression, alternate parsers, and non-file sources
  are not all implemented by generated ingestion. Start with the default text
  loader and inspect generated code before using these advanced combinations.
- Compose is a local development scaffold. Configure endpoints for container
  networking, credentials, readiness, and backups before deployment.
- AST checks prove Python syntax, not SDK/API compatibility or retrieval quality.

## Evaluation

Choose `evaluation.framework: ragas`, `deepeval`, or `both`. Supported metrics:
`faithfulness`, `answer_relevancy`, `context_precision`. Configure `judge_model`,
`pass_threshold`, and `num_test_cases`, then run `python eval.py` in the output
project. Supply `eval_dataset.json`, an array with `question` and `ground_truth`.
Context precision needs reference answers. Evaluation incurs provider API costs.
Malformed datasets and invalid or non-finite scores fail the evaluation gate.

## Development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,api]"
.venv/bin/ruff check .
.venv/bin/ruff format . --check
.venv/bin/mypy ragfactory
.venv/bin/pytest
```

CI runs static checks and the full suite on Python 3.11–3.14. Separate generated
runtime jobs install both frameworks on Python 3.12 and test real SDK retrieval
with local Qdrant and fake models, without paid API calls or model downloads.
An evaluation job checks Ragas scoring with intercepted provider calls, plus
Ragas and DeepEval imports and metric constructors.
Not every optional provider/store combination has this runtime coverage.

## Refresh evidence

- [Verification results and migration risks](research/MODERNIZATION_VERIFICATION_2026-10-20.md)
- [Model research and lifecycle notes](research/MODEL_REFRESH_2026-10-20.md)
- [SDK research and compatibility bands](research/SDK_REFRESH_2026-10-20.md)
- [Runtime findings](research/RUNTIME_REFRESH_2026-10-20.md)

The older [research library](research/INDEX.md) is historical background, not a
current guarantee for pricing, benchmarks, model availability, or implementation.

License metadata: Apache-2.0.
