# RAGBuilder — Revize Edilmiş Mimari Plan

> **Status**: Approved — implementation ready
> **Date**: April 5, 2026
> **Reviewed by**: Principal AI Architect
> **Evolution path**: CLI (Phase 1) → Web UI MVP (Phase 2) → Full Platform (Phase 3)

---

## Core Design Principle

**The code generator engine is the same across all phases.**
CLI, Web API, and UI are just different *interfaces* to the same core.
Phase 1 code is NOT throwaway — it becomes the backbone of Phase 2 and 3.

```
┌──────────────────────────────────────────────────────────┐
│                        CORE ENGINE                        │
│                                                           │
│  config.py          validator.py        generator.py      │
│  (Pydantic v2   →   (Compatibility  →   (Jinja2           │
│   Discriminated      Matrix +            Stage-based      │
│   Unions)            Cost Estimator)     Composition)     │
│                                                           │
│  versions.py        compatibility.py                      │
│  (Dependency         (Incompatible                        │
│   Manifest)          Pairs + Warnings)                    │
└────────────┬──────────────┬──────────────────────────────┘
             │              │              │
        ┌────┴───┐    ┌─────┴──┐    ┌─────┴─────┐
        │  CLI   │    │  API   │    │ Frontend  │
        │Phase 1 │    │Phase 2 │    │ Phase 3   │
        └────────┘    └────────┘    └───────────┘
```

---

## Final Project Structure

```
rag-automation/
│
├── ragbuilder/                        # pip install edilebilir Python paketi
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # Pydantic v2 models (Discriminated Unions)
│   │   ├── validator.py               # Compat check + cost estimator
│   │   ├── generator.py               # Jinja2 stage-based engine
│   │   ├── versions.py                # Dependency manifest (elle yönetilen)
│   │   └── compatibility.py           # Incompatible pairs + warning rules
│   │
│   ├── templates/
│   │   ├── stages/                    # Bileşen başına şablon (framework-agnostic)
│   │   │   ├── chunking/
│   │   │   │   ├── fixed.py.j2
│   │   │   │   ├── recursive.py.j2
│   │   │   │   ├── semantic.py.j2
│   │   │   │   ├── contextual.py.j2
│   │   │   │   ├── late.py.j2
│   │   │   │   └── page_level.py.j2
│   │   │   ├── embedding/
│   │   │   │   ├── openai.py.j2
│   │   │   │   ├── cohere.py.j2
│   │   │   │   ├── voyage.py.j2
│   │   │   │   ├── gemini.py.j2
│   │   │   │   ├── bge_m3.py.j2
│   │   │   │   └── nomic.py.j2
│   │   │   ├── vectordb/
│   │   │   │   ├── chromadb.py.j2
│   │   │   │   ├── qdrant.py.j2
│   │   │   │   ├── pinecone.py.j2
│   │   │   │   ├── weaviate.py.j2
│   │   │   │   ├── milvus.py.j2
│   │   │   │   └── pgvector.py.j2
│   │   │   ├── retrieval/
│   │   │   │   ├── dense.py.j2
│   │   │   │   ├── hybrid_rrf.py.j2
│   │   │   │   ├── hybrid_weighted.py.j2
│   │   │   │   ├── small_to_big.py.j2
│   │   │   │   └── sentence_window.py.j2
│   │   │   ├── reranker/
│   │   │   │   ├── cohere.py.j2
│   │   │   │   ├── cross_encoder.py.j2
│   │   │   │   ├── colbert.py.j2
│   │   │   │   └── flashrank.py.j2
│   │   │   └── llm/
│   │   │       ├── openai.py.j2
│   │   │       ├── anthropic.py.j2
│   │   │       ├── cohere.py.j2
│   │   │       └── ollama.py.j2
│   │   │
│   │   └── entrypoints/               # Stage'leri birleştiren pipeline şablonları
│   │       ├── langchain/
│   │       │   ├── pipeline.py.j2
│   │       │   └── ingestion.py.j2
│   │       ├── llamaindex/
│   │       │   ├── pipeline.py.j2
│   │       │   └── ingestion.py.j2
│   │       └── common/
│   │           ├── pyproject.toml.j2
│   │           ├── Dockerfile.j2
│   │           ├── docker-compose.yml.j2
│   │           ├── env.example.j2
│   │           └── README.md.j2
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py                    # Typer CLI
│
├── api/                               # Phase 2: FastAPI wrapper
│   ├── main.py
│   └── routers/
│       ├── pipelines.py
│       └── generate.py
│
├── frontend/                          # Phase 3: Next.js UI
│
├── tests/
│   ├── unit/
│   │   ├── test_config.py             # Phase 1a ile birlikte yazılır
│   │   ├── test_validator.py          # Phase 1b ile birlikte yazılır
│   │   ├── test_generator.py          # Phase 1b ile birlikte yazılır
│   │   └── test_templates.py          # Phase 1c ile birlikte yazılır
│   └── integration/
│       └── test_cli.py                # Phase 1d ile birlikte yazılır
│
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## Kritik Tasarım Kararları

### 1. Template Mimarisi: Stage-First, Compose-Later

388,800 kombinasyon için ayrı template yazılamaz.
Bunun yerine her bileşen (chunking, embedding, vectordb...) için
ayrı küçük template dosyaları yazılır.
Generator engine bunları seçip entrypoint'e inject eder.

```python
# generator.py mantığı:
def generate(config: RAGPipelineConfig) -> dict[str, str]:
    stages = {
        "chunking":  render("stages/chunking/{}.py.j2", config.indexing.chunking.type),
        "embedding": render("stages/embedding/{}.py.j2", config.indexing.embedding.provider),
        "vectordb":  render("stages/vectordb/{}.py.j2",  config.indexing.vector_db.provider),
        "retrieval": render("stages/retrieval/{}.py.j2", config.retrieval.search_type),
        "reranker":  render("stages/reranker/{}.py.j2",  config.post_retrieval.reranker.model),
        "llm":       render("stages/llm/{}.py.j2",       config.generation.llm.provider),
    }
    pipeline = render(
        f"entrypoints/{config.framework}/pipeline.py.j2",
        stages=stages, config=config
    )
    return {"pipeline.py": pipeline, ...}
```

### 2. Config Modeli: Pydantic v2 Discriminated Unions

Düz nested model yerine type discriminator kullanılır.
Bu sayede:
- Her search_type sadece kendi parametrelerini taşır
- JSON Schema'dan otomatik koşullu UI form üretilebilir (Phase 3)
- Validation hataları net ve anlamlı olur

```python
# YANLIŞ (düz model — tüm alanlar her zaman var)
class RetrievalConfig(BaseModel):
    search_type: str
    alpha: float        # sadece hybrid/weighted için gerekli
    rrf_k: int          # sadece hybrid/rrf için gerekli

# DOĞRU (discriminated union)
class DenseRetrievalConfig(BaseModel):
    type: Literal["dense"]
    top_k: int = 20

class HybridRRFConfig(BaseModel):
    type: Literal["hybrid_rrf"]
    top_k: int = 100
    rrf_k: int = 60

class HybridWeightedConfig(BaseModel):
    type: Literal["hybrid_weighted"]
    top_k: int = 100
    alpha: float = Field(0.5, ge=0.0, le=1.0)

RetrievalConfig = Annotated[
    Union[DenseRetrievalConfig, HybridRRFConfig, HybridWeightedConfig],
    Field(discriminator="type")
]
```

### 3. Dependency Manifest: Versions.py

"Auto-resolve" yok. Versiyonlar elle test edilmiş, merkezi bir
dosyada yönetilir. Her framework major versiyonunda güncellenir.

```python
# ragbuilder/core/versions.py
DEPENDENCY_MATRIX = {
    "langchain": {
        "base": ["langchain>=0.3.0,<0.4.0", "langchain-core>=0.3.0"],
        "embedding": {
            "openai":  ["langchain-openai>=0.2.0", "openai>=1.40.0"],
            "cohere":  ["langchain-cohere>=0.3.0", "cohere>=5.0.0"],
            "voyage":  ["voyageai>=0.3.0"],
            "bge_m3":  ["FlagEmbedding>=1.2.0", "torch>=2.0.0"],
        },
        "vectordb": {
            "qdrant":   ["langchain-qdrant>=0.2.0", "qdrant-client>=1.7.0"],
            "pinecone": ["langchain-pinecone>=0.2.0", "pinecone-client>=3.0.0"],
            "chromadb": ["langchain-chroma>=0.1.0", "chromadb>=0.5.0"],
            "weaviate": ["langchain-weaviate>=0.0.3", "weaviate-client>=4.0.0"],
            "pgvector": ["langchain-postgres>=0.0.9", "psycopg[binary]>=3.1.0"],
        },
        "reranker": {
            "cohere":        ["cohere>=5.0.0"],
            "cross_encoder": ["sentence-transformers>=3.0.0"],
            "colbert":       ["ragatouille>=0.0.8"],
            "flashrank":     ["flashrank>=0.2.0"],
        },
    },
    "llamaindex": {
        "base": ["llama-index>=0.11.0", "llama-index-core>=0.11.0"],
        # ... benzer yapı
    }
}
```

### 4. Compatibility Matrix: Ayrı Data Structure

Validator içine gömmek yerine ayrı dosya.
CLI wizard, API validator ve Phase 3 UI'ı hepsi bunu kullanır.

```python
# ragbuilder/core/compatibility.py

INCOMPATIBLE = [
    # (bileşen_a, bileşen_b, hata_mesajı)
    ("generation.flare",    "llm.anthropic",
     "FLARE requires token-level logprobs. Anthropic API does not expose logprobs."),

    ("generation.self_rag", "*",
     "Self-RAG requires a fine-tuned model. Select 'self-rag-llama2-7b' as LLM instead."),

    ("retrieval.graphrag",  "latency.realtime",
     "GraphRAG global search averages 30-120s. Not suitable for real-time use."),

    ("embedding.bge_m3",    "hosting.cloud_only",
     "BGE-M3 requires self-hosting. Add Docker/GPU to your infrastructure."),
]

WARNINGS = [
    # (koşul, uyarı, cost_formula)
    ("chunking.contextual",
     "Contextual chunking costs ~$1.02/M doc tokens (with prompt caching). "
     "Estimated for your corpus: ${estimated_cost}",
     lambda cfg: estimate_contextual_cost(cfg)),

    ("retrieval.graphrag",
     "GraphRAG indexing requires ~${estimated_cost} in LLM calls for your corpus size.",
     lambda cfg: estimate_graphrag_cost(cfg)),

    ("embedding.provider != indexing.embedding.provider",
     "Query embedding model differs from index embedding model. This will degrade retrieval quality.",
     None),
]
```

### 5. Generated Code Validation

Jinja2 string üretir — syntax hatası içerebilir.
Her generate işleminden sonra zorunlu kontrol:

```python
# generator.py içinde
import ast

def validate_generated_code(code: str, filename: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise GeneratorError(
            f"Generated {filename} has syntax error at line {e.lineno}: {e.msg}\n"
            f"This is a bug in RAGBuilder templates. Please report it."
        )
```

### 6. YAML Round-Trip: First-Class Feature

Config hem kaydedilir hem yüklenir. Git-friendly.

```bash
# Üret + config'i kaydet
ragbuilder init --name "my-rag" --chunking contextual \
  --embedding voyage --vectordb qdrant \
  --save-config pipeline.yaml

# Config'den üret (idempotent)
ragbuilder generate --config pipeline.yaml --output ./my-rag

# Sadece validate et
ragbuilder validate --config pipeline.yaml

# Seçenekleri listele
ragbuilder options
ragbuilder options --component embedding
```

---

## Phase 1: CLI Tool — Detaylı Sub-Fazlar

**Kural: Her sub-fazda hem kod hem testi birlikte yazılır. Bir sonraki faza geçmeden testler geçmeli.**

### Phase 1a — Foundation (5 dosya)

| Dosya | İçerik |
|-------|--------|
| `pyproject.toml` | Paket tanımı, Typer/Pydantic/Jinja2 bağımlılıkları, `ragbuilder` entry point |
| `ragbuilder/__init__.py` | Version string |
| `ragbuilder/core/config.py` | Tüm Pydantic v2 modelleri (discriminated unions) |
| `ragbuilder/core/versions.py` | Dependency manifest |
| `tests/unit/test_config.py` | Config model testleri (valid/invalid configs, serialization) |

**Verification:** `pytest tests/unit/test_config.py` → tüm testler geçmeli

---

### Phase 1b — Validator + Generator Engine (4 dosya)

| Dosya | İçerik |
|-------|--------|
| `ragbuilder/core/compatibility.py` | INCOMPATIBLE + WARNINGS data structures |
| `ragbuilder/core/validator.py` | Compat check + cost estimator (compatibility.py'ı kullanır) |
| `ragbuilder/core/generator.py` | Jinja2 stage-based engine + AST validation |
| `tests/unit/test_validator.py` | Compat hata senaryoları, cost estimation |

**Verification:** `pytest tests/unit/test_validator.py` → geçmeli

---

### Phase 1c — Templates (6 dosya ilk set + test)

İlk iterasyonda en çok kullanılan kombinasyon için template'ler:
- Chunking: `recursive.py.j2`, `contextual.py.j2`
- Embedding: `openai.py.j2`, `voyage.py.j2`
- VectorDB: `chromadb.py.j2`, `qdrant.py.j2`
- Retrieval: `dense.py.j2`, `hybrid_rrf.py.j2`
- LLM: `openai.py.j2`, `anthropic.py.j2`
- Entrypoint: `langchain/pipeline.py.j2`, `common/pyproject.toml.j2`
- `tests/unit/test_templates.py` — üretilen kodun AST parse edildiğini test et

**Verification:** `pytest tests/unit/test_templates.py` → geçmeli

---

### Phase 1d — CLI (3 dosya)

| Dosya | İçerik |
|-------|--------|
| `ragbuilder/cli/__init__.py` | Boş init |
| `ragbuilder/cli/main.py` | Typer CLI: `init`, `generate`, `validate`, `options` komutları |
| `tests/integration/test_cli.py` | End-to-end: `ragbuilder init ...` → dosyalar üretildi mi? |

**Verification:**
```bash
pip install -e .
ragbuilder --help                     # çalışmalı
ragbuilder options                    # listelenmeli
ragbuilder validate --config ...      # hata yoksa OK
ragbuilder generate --config ... --output /tmp/test-rag
python -c "import ast; ast.parse(open('/tmp/test-rag/pipeline.py').read())"  # syntax OK
```

---

## Phase 2: Web UI MVP

Core engine değişmez. Yeni bileşenler:

| Bileşen | Teknoloji | İş |
|---------|----------|-----|
| REST API | FastAPI | `ragbuilder.core`'u HTTP üzerinden expose eder |
| Frontend | Next.js + shadcn/ui | JSON Schema → otomatik formlar |
| DB | SQLite (local) → PostgreSQL (prod) | Pipeline kayıt/listeleme |
| Auth | Yok (local MVP) → NextAuth.js (prod) | Phase 3'e ertelendi |

**API endpoints:**
```
POST   /api/pipelines           → config kaydet
GET    /api/pipelines/{id}      → config yükle
POST   /api/pipelines/validate  → compat check + cost estimate
POST   /api/pipelines/{id}/generate → zip üret + indir
GET    /api/options             → tüm bileşenler + parametreler
GET    /api/options/{component} → tek bileşen seçenekleri
```

---

## Phase 3: Full Platform

- User auth (NextAuth.js)
- Pipeline registry (versioning, fork, share)
- Visual pipeline canvas (React Flow)
- Sandbox deploy (üretilen kodu Docker'da çalıştır)
- RAGAS eval gate (deploy öncesi kalite kontrol)
- Production deploy (K8s / serverless)
- Monitoring dashboard (query logs, quality drift, cost tracking)

---

## Technology Stack

| Katman | Teknoloji | Neden |
|--------|----------|-------|
| CLI | Typer | Type-annotated, auto `--help`, Click üzerine |
| Config | Pydantic v2 | Validation + JSON Schema export (Phase 3 için) |
| Code generation | Jinja2 | Battle-tested, okunabilir template'ler |
| Packaging | pyproject.toml + hatchling | Modern Python, pip install edilebilir |
| Phase 2 API | FastAPI | Async, Pydantic native, OpenAPI auto-gen |
| Phase 3 UI | Next.js + shadcn/ui | JSON Schema → react-jsonschema-form |
| Testing | pytest | Standard, Typer CliRunner entegrasyonu var |

---

## Başlangıç: Onaylanan Kombinasyonlar (MVP Set)

Phase 1c'de ilk üretilecek template kombinasyonları.
Bu 4 kombinasyon ile %80 use case karşılanır:

| İsim | Chunking | Embedding | VectorDB | Search | LLM |
|------|----------|-----------|----------|--------|-----|
| **quick-start** | recursive | openai-small | chromadb | dense | openai |
| **production-standard** | recursive | voyage | qdrant | hybrid_rrf | claude-sonnet |
| **high-accuracy** | contextual | voyage | qdrant | hybrid_rrf+rerank | claude-sonnet |
| **local-dev** | recursive | bge-m3 | chromadb | hybrid_rrf | ollama |
