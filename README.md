<div align="center">

# ragfactory

**Generate production-ready RAG pipelines from a single config file.**

[![PyPI version](https://img.shields.io/pypi/v/ragfactory?color=blue)](https://pypi.org/project/ragfactory/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![LangChain](https://img.shields.io/badge/LangChain-%E2%9C%93-blue)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-%E2%9C%93-purple)](https://www.llamaindex.ai/)

</div>

---

You have a document corpus. You need a production-grade RAG system. You don't want to spend three days wiring together chunkers, embedders, vector DBs, rerankers, and LLMs — only to discover on day four that you chose the wrong retrieval strategy.

**ragfactory solves this.** Describe your pipeline in a single YAML file. Get back a fully-wired, Dockerised Python project with every component integrated, validated, and ready to run.

```bash
pip install ragfactory
ragfactory init --name my-rag --vector-db qdrant --llm anthropic --output ./my-rag
```

That's it. Eight files. Zero boilerplate. Ship in minutes, not days.

---

## What gets generated

Every `ragfactory generate` produces a complete, standalone Python project:

| File | Description |
|------|-------------|
| `pipeline.py` | Query pipeline — retrieval, reranking, generation, fully wired |
| `ingestion.py` | Indexing pipeline — load, chunk, embed, upsert |
| `config.yaml` | Serialized, reloadable copy of your exact config |
| `pyproject.toml` | Pinned dependencies for every chosen component |
| `.env.example` | Required API keys, pre-filled with the right variable names |
| `Dockerfile` | Container-ready, production base image |
| `docker-compose.yml` | Spins up vector DB + app in one command |
| `README.md` | Component-specific setup guide for your exact combination |

The generated code is **not a wrapper**. It's idiomatic Python that calls the upstream SDKs directly — no ragfactory runtime dependency at execution time.

---

## Installation

```bash
pip install ragfactory
```

**Requirements:** Python 3.11+

---

## Quick start

### Option A — Interactive wizard

```bash
ragfactory init \
  --name customer-support-rag \
  --vector-db qdrant \
  --embedding voyage \
  --llm anthropic \
  --output ./customer-support-rag
```

### Option B — Config-driven generation

```yaml
# pipeline.yaml
name: customer-support-rag
framework: langchain

indexing:
  chunking:
    type: contextual          # LLM-prepended context — +49-67% recall (Anthropic, 2024)
    chunk_size: 512
    chunk_overlap: 50
    context_model: gpt-4o-mini
  embedding:
    type: voyage              # Best MTEB 2024 retrieval benchmarks
    model: voyage-3-large
  vector_db:
    type: qdrant              # 8500-12000 QPS, best metadata filtering
    url: http://localhost:6333
    collection_name: support-docs

retrieval:
  type: hybrid_rrf            # BM25 + dense via Reciprocal Rank Fusion — +15-30% recall
  top_k: 20
  rrf_k: 60

post_retrieval:
  reranker:
    type: cohere              # API reranker, no GPU required
    top_n: 5

generation:
  llm:
    type: anthropic
    model: claude-sonnet-4-6
    temperature: 0.1
```

```bash
ragfactory generate --config pipeline.yaml --output ./customer-support-rag
cd customer-support-rag
pip install -e .
```

### Option C — Save config, generate later

```bash
# Scaffold the YAML without generating code
ragfactory init --name my-rag --save-config ./my-rag.yaml

# Edit my-rag.yaml, then generate
ragfactory generate --config ./my-rag.yaml --output ./my-rag
```

---

## CLI reference

### `ragfactory generate`

Generate a complete pipeline project from a config file.

```bash
ragfactory generate --config pipeline.yaml --output ./my-pipeline

# Skip compatibility validation (advanced use)
ragfactory generate --config pipeline.yaml --output ./my-pipeline --force
```

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file (required) |
| `--output` | Output directory (default: config `name` field) |
| `--force` | Skip compatibility validation, generate anyway |

---

### `ragfactory validate`

Validate a config file without generating anything. Surfaces incompatibilities, warnings, and cost alerts.

```bash
ragfactory validate --config pipeline.yaml
```

**Exit codes:**
- `0` — Config is valid (warnings may still appear)
- `1` — Config has incompatible components; generation is blocked

**Example output:**

```
✓  Config is valid

  WARN_CHROMADB        ChromaDB is limited to ~7M vectors — prototyping only
  WARN_CONTEXTUAL      Contextual chunking costs ~$1.02/M input tokens
```

---

### `ragfactory init`

Scaffold a new pipeline with smart defaults and automatic compatibility enforcement.

```bash
ragfactory init \
  --name my-pipeline \
  --framework langchain \
  --vector-db qdrant \
  --embedding openai \
  --chunking recursive \
  --retrieval hybrid_rrf \
  --reranker cohere \
  --llm anthropic \
  --output ./my-pipeline
```

**Auto-retrieval logic:** ragfactory picks the best retrieval strategy for your vector DB automatically.
- `chromadb` / `pinecone` → `dense` (no sparse index support)
- `qdrant` / `weaviate` / `milvus` / `pgvector` → `hybrid_rrf`

You can always override with `--retrieval`.

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | required | Pipeline name (lowercase, alphanumeric, hyphens) |
| `--framework` | `langchain` | `langchain` or `llamaindex` |
| `--vector-db` | `qdrant` | Vector database |
| `--embedding` | `openai` | Embedding model provider |
| `--chunking` | `recursive` | Chunking strategy |
| `--retrieval` | auto | Retrieval strategy |
| `--reranker` | none | Reranker (optional) |
| `--llm` | `openai` | LLM provider |
| `--output` | `./name` | Output directory |
| `--save-config` | — | Write YAML only, skip code generation |

---

### `ragfactory options`

Explore all available components and their descriptions.

```bash
# All components
ragfactory options

# Filter by category
ragfactory options --component embedding
ragfactory options --component retrieval

# Machine-readable JSON
ragfactory options --json
```

**Available categories:** `chunking`, `embedding`, `vectordb`, `retrieval`, `reranker`, `llm`

---

## Component matrix

### Chunking strategies

| Strategy | Best for | Notes |
|----------|----------|-------|
| `recursive` | General purpose | Production default. Hierarchical separators. |
| `fixed` | Uniform corpora | Simple baseline. Fixed token windows. |
| `semantic` | Topic-shifting docs | Embedding-based breakpoints. Handles topic drift. |
| `contextual` | High-recall pipelines | LLM-prepended context per chunk. **+49–67% recall** (Anthropic, 2024). Costs ~$1.02/M tokens. |
| `late` | Long-form retrieval | Jina Late Chunking. Token-level pooling after full-doc encoding. Requires Jina embeddings. |
| `page_level` | Structured PDFs | One chunk per page. Good for dense reference material. |
| `sentence_window` | Fine-grained Q&A | Retrieve at sentence level, expand to surrounding window. |

### Embedding models

| Model | Dims | Highlights |
|-------|------|------------|
| `openai` | 1536 | `text-embedding-3-small/large`. Fast, reliable, great default. |
| `cohere` | 1024 | `embed-v4.0`. Multilingual. Separate input types for query/document. |
| `voyage` | 1024 | `voyage-3-large`. **Top MTEB 2024 retrieval benchmarks.** |
| `gemini` | 768 | `text-embedding-004`. Google ecosystem integration. |
| `bge_m3` | 1024 | `BAAI/BGE-M3`. Self-hosted. Dense + sparse + ColBERT in one model. |
| `nomic` | 768 | `nomic-embed-text-v1.5`. Self-hosted or API. Configurable dims: 64–768. |
| `jina` | 1024 | `jina-embeddings-v3`. 8192-token context. Native late chunking support. |

### Vector databases

| DB | Scale | Highlights |
|----|-------|------------|
| `chromadb` | < 7M vectors | Embedded, in-process. Zero infra. Prototyping only. |
| `qdrant` | 100M+ | **Production default.** Rust. 8500–12000 QPS. Best metadata filtering. |
| `pinecone` | 1.4B+ | Serverless. Zero-ops. Pay-per-query. |
| `weaviate` | 100M+ | Native hybrid search. Multi-tenancy. GraphQL + REST. |
| `milvus` | 1B+ | Distributed. GPU-accelerated indexing (CAGRA). Enterprise-grade. |
| `pgvector` | 10M+ | PostgreSQL extension. Best for teams with an existing Postgres stack. |

### Retrieval strategies

| Strategy | Recall delta | Notes |
|----------|-------------|-------|
| `dense` | Baseline | Pure vector similarity. Fastest. |
| `hybrid_rrf` | **+15–30%** | BM25 + dense via Reciprocal Rank Fusion. Keyword + semantic. |
| `hybrid_weighted` | +10–25% | BM25 + dense with tunable alpha weight. More control than RRF. |
| `small_to_big` | +5–15% | Retrieve child chunks, return parent context. Better answer coherence. |
| `sentence_window` | +5–10% | Retrieve sentence, expand to window. Fine-grained + coherent. |

> **Compatibility note:** `hybrid_rrf` and `hybrid_weighted` require sparse vector support. ChromaDB and Pinecone fall back to `dense` automatically.

### Rerankers

| Reranker | GPU required | Notes |
|----------|-------------|-------|
| `cohere` | No | Cohere Rerank API. Fast, production-ready, no infra. |
| `cross_encoder` | Recommended | Best quality. Cross-attention scoring. GPU needed for production throughput. |
| `colbert` | No | RAGatouille ColBERT. Token-level interaction. Requires 6–10× disk space. |
| `flashrank` | No | Fastest local reranker. CPU-friendly. Best for latency-first setups. |

### LLM providers

| Provider | Models | Default |
|----------|--------|---------|
| `openai` | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | `gpt-4o-mini` |
| `anthropic` | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 | `claude-sonnet-4-6` |
| `cohere` | command-r-plus, command-r, command | Grounding-optimised, inline citation support |
| `ollama` | Any Ollama model | Local inference, zero API cost |

---

## Frameworks

Both **LangChain** and **LlamaIndex** are fully supported. The generated code uses each framework's native APIs and idioms — not a shared abstraction layer on top of them.

```yaml
framework: langchain    # or: llamaindex
```

| Feature | LangChain | LlamaIndex |
|---------|:---------:|:----------:|
| LCEL pipeline chains | ✓ | — |
| EnsembleRetriever (hybrid) | ✓ | ✓ |
| Small-to-big retrieval | ✓ | ✓ |
| Sentence Window Retrieval (native) | — | ✓ |
| Node post-processors | — | ✓ |
| Streaming support | ✓ | ✓ |

---

## Full config reference

```yaml
name: my-pipeline                    # required; lowercase, alphanumeric, hyphens; 1-64 chars
version: "1.0"                       # config schema version (default: "1.0")
framework: langchain                 # langchain | llamaindex

# ── Indexing ──────────────────────────────────────────────────────────────────
indexing:
  chunking:
    type: recursive                  # fixed | recursive | semantic | contextual | late | page_level | sentence_window
    chunk_size: 512                  # tokens per chunk
    chunk_overlap: 50                # overlap between chunks

  embedding:
    type: openai                     # openai | cohere | voyage | gemini | bge_m3 | nomic | jina
    model: text-embedding-3-small    # provider-specific model name (optional)

  vector_db:
    type: qdrant                     # chromadb | qdrant | pinecone | weaviate | milvus | pgvector
    url: http://localhost:6333       # for qdrant, weaviate, milvus
    collection_name: my-collection

# ── Pre-retrieval (optional) ──────────────────────────────────────────────────
pre_retrieval:
  query_rewriting:
    enabled: true
    strategy: multi_query            # multi_query | sub_question | step_back
    num_rewrites: 3
  hyde:
    enabled: true
    num_hypotheses: 3

# ── Retrieval ─────────────────────────────────────────────────────────────────
retrieval:
  type: hybrid_rrf                   # dense | hybrid_rrf | hybrid_weighted | small_to_big | sentence_window
  top_k: 20
  rrf_k: 60                          # hybrid_rrf only (default: 60)

# ── Post-retrieval (optional) ─────────────────────────────────────────────────
post_retrieval:
  reranker:
    type: cohere                     # cohere | cross_encoder | colbert | flashrank
    top_n: 5
  context_assembly:
    ordering: relevance_first        # relevance_first | chronological | reverse_relevance
    max_sources: 5
    source_attribution: true

# ── Generation ────────────────────────────────────────────────────────────────
generation:
  llm:
    type: anthropic                  # openai | anthropic | cohere | ollama
    model: claude-sonnet-4-6
    temperature: 0.1
    max_tokens: 1024

# ── Evaluation (optional) ─────────────────────────────────────────────────────
evaluation:
  framework: ragas                   # ragas | deepeval | both
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
  num_test_cases: 50
  pass_threshold: 0.7
```

---

## Compatibility validation

ragfactory validates every config before generating code. Incompatible combinations are blocked with a clear error code. Warnings surface cost and performance risks without blocking generation.

### Blocked combinations

| Error code | Meaning |
|------------|---------|
| `INCOMPAT_HYBRID_RRF_CHROMADB` | ChromaDB has no sparse index — hybrid RRF requires it |
| `INCOMPAT_HYBRID_WEIGHTED_CHROMADB` | Same constraint for weighted hybrid |
| `INCOMPAT_HYBRID_RRF_PINECONE` | Hybrid RRF not reliably exposed in Pinecone integrations |
| `INCOMPAT_SENTENCE_WINDOW_LANGCHAIN` | Sentence Window Retrieval is LlamaIndex-native |
| `INCOMPAT_LATE_OPENAI` | Late chunking requires Jina embeddings, not OpenAI |
| `INCOMPAT_LATE_BGE_M3` | Late chunking requires Jina embeddings, not BGE-M3 |
| `INCOMPAT_FLARE_ANTHROPIC` | FLARE requires LLM logprobs; Anthropic doesn't expose them |
| `INCOMPAT_FLARE_COHERE_LLM` | FLARE incompatible with Cohere LLM (no logprobs) |
| `LATE_CHUNKING_FLAG_NOT_SET` | `chunking.type=late` but `embedding.late_chunking=false` |

### Warnings

| Warning code | Meaning |
|--------------|---------|
| `WARN_CHROMADB` | ChromaDB caps at ~7M vectors — prototyping only |
| `WARN_CONTEXTUAL` | Contextual chunking costs ~$1.02/M input tokens |
| `WARN_CROSS_ENCODER` | Cross-encoder reranker needs GPU for production throughput |
| `WARN_BGE_M3` | BGE-M3 self-hosted; CPU inference is 10–50× slower than GPU |
| `WARN_HYDE` | HyDE adds 1 extra LLM call per query (~100ms extra latency) |
| `WARN_COLBERT` | ColBERT requires 6–10× the disk space of a dense index |
| `WARN_CHROMADB` | ChromaDB is not suited for > 7M vectors |
| `CONTEXTUAL_CHUNKING_EXTRA_API_KEY` | Context model uses a different provider — extra API key required |
| `RERANKER_TOP_N_EXCEEDS_TOP_K` | `reranker.top_n >= retrieval.top_k` — likely misconfiguration |

---

## Four validated starting points

### Prototype — Zero infrastructure
Runs locally with no external services. Great for experimentation.

```yaml
name: prototype
framework: langchain
indexing:
  chunking: {type: recursive}
  embedding: {type: openai}
  vector_db: {type: chromadb}
retrieval: {type: dense}
generation:
  llm: {type: openai}
```

### Production — Best performance per dollar
Full hybrid retrieval, reranking, contextual chunking.

```yaml
name: production
framework: langchain
indexing:
  chunking: {type: contextual, context_model: gpt-4o-mini}
  embedding: {type: voyage, model: voyage-3-large}
  vector_db: {type: qdrant, url: http://localhost:6333, collection_name: docs}
retrieval: {type: hybrid_rrf, top_k: 20}
post_retrieval:
  reranker: {type: cohere, top_n: 5}
generation:
  llm: {type: anthropic, model: claude-sonnet-4-6}
```

### Serverless — Zero operations
Fully managed, infinitely scalable. Pay-per-query with no infra to maintain.

```yaml
name: serverless
framework: langchain
indexing:
  chunking: {type: recursive}
  embedding: {type: openai}
  vector_db: {type: pinecone, index_name: my-index, environment: us-east-1}
retrieval: {type: dense, top_k: 10}
generation:
  llm: {type: openai, model: gpt-4o}
```

### Air-gapped — Fully local
Zero API calls. Everything runs on your hardware. Privacy-first.

```yaml
name: local
framework: langchain
indexing:
  chunking: {type: recursive}
  embedding: {type: bge_m3}
  vector_db: {type: chromadb}
retrieval: {type: dense}
generation:
  llm: {type: ollama, model: llama3.2, base_url: http://localhost:11434}
```

---

## Environment setup

Each generated project includes a `.env.example` pre-filled for your exact component selection. Copy it, fill in your keys, and run.

```bash
cp .env.example .env
```

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | openai embedding or LLM |
| `ANTHROPIC_API_KEY` | anthropic LLM |
| `COHERE_API_KEY` | cohere embedding, reranker, or LLM |
| `VOYAGE_API_KEY` | voyage embedding |
| `PINECONE_API_KEY` | pinecone vector DB |
| `WEAVIATE_API_KEY` | weaviate cloud |
| `GOOGLE_API_KEY` | gemini embedding |

---

## Architecture

```
ragfactory/
├── models/         Pydantic v2 config schemas — every component, every option
├── core/
│   ├── validator.py    Cross-field compatibility rules (errors + warnings + cost alerts)
│   └── generator.py    Jinja2 template renderer → fully-wired Python project
├── templates/
│   ├── stages/         One .j2 template per component (42 templates total)
│   │   ├── chunking/   7 strategies
│   │   ├── embedding/  7 providers
│   │   ├── vectordb/   6 databases
│   │   ├── retrieval/  5 strategies
│   │   ├── reranker/   4 rerankers
│   │   └── llm/        4 providers
│   └── entrypoints/
│       ├── langchain/  pipeline.py.j2, ingestion.py.j2
│       ├── llamaindex/ pipeline.py.j2, ingestion.py.j2
│       └── common/     Dockerfile.j2, pyproject.toml.j2, .env.example.j2, README.md.j2
└── cli/
    └── main.py         Typer CLI: generate, validate, init, options
```

**Possible combinations:** 7 chunking × 7 embedding × 6 vector DB × 5 retrieval × 5 reranker × 4 LLM × 2 frameworks = **58,800 unique pipeline configurations.**

---

## Development

```bash
git clone https://github.com/ragfactory/ragfactory
cd ragfactory
pip install -e ".[dev]"
```

**Run tests:**

```bash
pytest                         # full suite
pytest tests/unit/             # unit tests only
pytest tests/integration/      # CLI integration tests (real filesystem)
pytest --cov=ragfactory        # with coverage report
```

**Lint and type-check:**

```bash
ruff check .
mypy ragfactory/
```

**Verify a generated project's Python is syntactically valid:**

```bash
ragfactory generate --config tests/fixtures/quick_start.yaml --output /tmp/test-out
python -c "import ast; ast.parse(open('/tmp/test-out/pipeline.py').read()); print('OK')"
```

---

## Roadmap

- [x] Phase 1 — CLI: `generate`, `validate`, `init`, `options`
- [ ] Phase 2 — REST API (FastAPI): programmatic pipeline generation
- [ ] Phase 3 — Web UI: visual pipeline builder with live validation
- [ ] Advanced generation techniques: CRAG, FLARE, Agentic RAG
- [ ] Evaluation harness: RAGAS and DeepEval integration
- [ ] Ingestion sources: S3, URLs, Google Drive, Notion
- [ ] LangGraph multi-agent pipeline support
- [ ] VS Code extension: config autocomplete + inline validation

---

## License

Apache 2.0 — free for commercial and private use.

---

<div align="center">

Built by developers who got tired of wiring the same RAG stack from scratch for the fifth time.

</div>
