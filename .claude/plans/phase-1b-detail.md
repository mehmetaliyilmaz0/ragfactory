# Phase 1b — Detailed Implementation Plan

> **Status**: Approved — ready for implementation
> **Date**: April 5, 2026
> **Authored by**: Principal AI Architect + Lead AI Researcher
> **Depends on**: Phase 1a complete (102/102 tests passing)
> **Delivers**: `compatibility.py` → `validator.py` → `generator.py` + full test coverage

---

## Table of Contents

1. [compatibility.py Design](#1-compatibilitypy)
2. [validator.py Design](#2-validatorpy)
3. [generator.py Design](#3-generatorpy)
4. [Test Strategy](#4-test-strategy)
5. [Implementation Order](#5-implementation-order)
6. [Risk Register](#6-risk-register)

---

## 1. compatibility.py

### 1.1 Design Philosophy

**Pure data module. Zero logic. Zero imports from project code.**

This is the single source of truth for all compatibility knowledge.
CLI, validator, Phase 2 API, and Phase 3 UI all read from here.
It must be importable in isolation — no circular dependency risk.

### 1.2 Data Structures

```python
from dataclasses import dataclass
from enum import Enum

class Severity(str, Enum):
    INFO        = "info"         # informational, no action needed
    WARNING     = "warning"      # may cause issues or suboptimal performance
    COST_ALERT  = "cost_alert"   # significant cost implication

@dataclass(frozen=True)
class IncompatiblePair:
    component_a: str        # dot-path, e.g. "generation.advanced.flare"
    component_b: str        # dot-path or "*" for unconditional
    reason: str             # human-readable error
    doc_url: str | None     # optional upstream issue/docs link

@dataclass(frozen=True)
class CompatibilityWarning:
    condition: str                  # dot-path or compound identifier
    message: str                    # may contain {placeholders}
    severity: Severity
    cost_per_million: float | None  # USD/1M tokens; None if not cost-related

@dataclass(frozen=True)
class CrossFieldRule:
    rule_id: str
    description: str
    # Logic lives in validator.py; this is metadata only
```

Why `@dataclass(frozen=True)` not Pydantic: these are compile-time constants,
not user input. Frozen dataclasses signal "this data does not change at runtime."
Keeps compatibility.py Pydantic-free — importable anywhere without the overhead.

### 1.3 INCOMPATIBLE — Complete List

| # | component_a | component_b | Reason |
|---|---|---|---|
| 1 | `generation.advanced.flare` | `generation.llm.anthropic` | FLARE requires token-level logprobs. Anthropic API does not expose logprobs. |
| 2 | `generation.advanced.flare` | `generation.llm.cohere_llm` | Cohere Command-R does not expose per-token logprobs in its API. |
| 3 | `generation.advanced.flare` | `generation.llm.ollama` | Ollama logprob support is model-dependent and not reliably surfaced by langchain-ollama / llama-index-llms-ollama integrations. |
| 4 | `indexing.chunking.late` | `indexing.embedding.openai` | Late chunking is a Jina-specific architecture. Requires `jina-embeddings-v3` with `late_chunking=True`. |
| 5 | `indexing.chunking.late` | `indexing.embedding.cohere` | Same as #4. |
| 6 | `indexing.chunking.late` | `indexing.embedding.voyage` | Same as #4. |
| 7 | `indexing.chunking.late` | `indexing.embedding.gemini` | Same as #4. |
| 8 | `indexing.chunking.late` | `indexing.embedding.bge_m3` | Same as #4. |
| 9 | `indexing.chunking.late` | `indexing.embedding.nomic` | Same as #4. |
| 10 | `retrieval.hybrid_rrf` | `indexing.vector_db.chromadb` | ChromaDB has no native sparse/BM25 support. Phase 1c templates do not implement external BM25 for ChromaDB. |
| 11 | `retrieval.hybrid_weighted` | `indexing.vector_db.chromadb` | Same as #10. |
| 12 | `retrieval.hybrid_rrf` | `indexing.vector_db.pinecone` | langchain-pinecone / llamaindex-pinecone integrations do not reliably expose Pinecone's sparse-dense API. Blocked until integration matures. |
| 13 | `retrieval.hybrid_weighted` | `indexing.vector_db.pinecone` | Same as #12. |
| 14 | `retrieval.sentence_window` | `framework.langchain` | SentenceWindowNodeParser + MetadataReplacementPostProcessor are LlamaIndex-native. No direct LangChain equivalent. |

### 1.4 CROSS_FIELD_RULES — Complete List

These require inspecting field *values*, not just types.
Metadata only — logic implemented in validator.py.

| rule_id | description |
|---|---|
| `late_chunking_jina_v3_flag` | When `chunking.type=="late"`: embedding must be `jina`, `model=="jina-embeddings-v3"`, `late_chunking==True`. Three separate conditions, all required. |
| `late_chunking_jina_v2` | `jina-embeddings-v2-base-en` does not support late chunking (8K context window required). ERROR. |
| `contextual_chunking_throughput` | When `chunking.type=="contextual"` and `context_model` is an Ollama-style local model: feasible but extremely slow at scale (~100x vs Claude Haiku). WARNING. |
| `contextual_extra_api_key` | When `chunking.type=="contextual"` and `context_model` differs from main LLM provider: extra API key required beyond pipeline LLM. INFO. |
| `multiple_advanced_techniques` | At most one of `crag`, `flare`, `agentic` should be enabled. Pydantic does not enforce this. WARNING if >1 enabled. |
| `reranker_top_n_vs_top_k` | `reranker.top_n >= retrieval.top_k` means reranker cannot select more than was retrieved — almost always a user mistake. WARNING. |

### 1.5 WARNINGS — Complete List

| # | condition | message | severity | cost_per_million |
|---|---|---|---|---|
| 1 | `chunking.type == "contextual"` | Contextual chunking costs ~$1.02/M doc tokens (with prompt caching; $5.10/M without). | COST_ALERT | 1.02 |
| 2 | `chunking.type == "proposition"` | Proposition chunking: one LLM call per paragraph. ~$2.50/M tokens with gpt-4o-mini. | COST_ALERT | 2.50 |
| 3 | `generation.advanced.agentic.enabled` | Agentic RAG: up to `max_reasoning_steps` LLM calls per query. Expect 3–10× query cost vs simple RAG. | COST_ALERT | None |
| 4 | `generation.advanced.crag.web_search_fallback` | CRAG web search adds 1–3s latency per low-confidence query. Disable in air-gapped / latency-sensitive deployments. | WARNING | None |
| 5 | `indexing.embedding.bge_m3` | BGE-M3 is self-hosted. CPU inference is 10–50× slower than GPU. Ensure deployment has GPU. | WARNING | None |
| 6 | `post_retrieval.reranker.cross_encoder` | Cross-encoder reranker requires GPU for >10 QPS. CPU viable only at low throughput. | WARNING | None |
| 7 | `post_retrieval.reranker.colbert` | ColBERT (RAGatouille) index is 6–10× document size after ColBERTv2 compression. Plan storage accordingly. | INFO | None |
| 8 | `pre_retrieval.hyde.enabled` | HyDE adds one LLM call per query for hypothesis generation. Small latency + cost increase. | INFO | None |
| 9 | `indexing.vector_db.chromadb` | ChromaDB is embedded/prototyping-only. Max tested: ~7M vectors. Not recommended for production. | WARNING | None |
| 10 | `evaluation is not None` | RAGAS/DeepEval eval with judge model: ~$2–5 per 50-sample run with GPT-4o. | COST_ALERT | None |

---

## 2. validator.py

### 2.1 Output Types

```python
from enum import Enum
from dataclasses import dataclass, field

class ValidationSeverity(str, Enum):
    ERROR   = "error"    # pipeline will not function
    WARNING = "warning"  # may cause issues / suboptimal
    INFO    = "info"     # informational
    COST    = "cost"     # cost estimation

@dataclass
class ValidationIssue:
    severity:       ValidationSeverity
    code:           str            # machine-readable, e.g. "FLARE_NO_LOGPROBS"
    message:        str
    component_path: str            # dot-path to offending field
    suggestion:     str | None = None

@dataclass
class CostEstimate:
    component:              str
    description:            str
    cost_per_million_tokens: float
    estimated_total:        float | None  # None if corpus_tokens not provided

@dataclass
class ValidationResult:
    valid:  bool                     # True iff zero ERRORs
    issues: list[ValidationIssue]   = field(default_factory=list)
    costs:  list[CostEstimate]      = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
```

Why dataclasses not Pydantic: `ValidationResult` is output, not user input.
Pydantic adds validation overhead on output types for no benefit. Plain
dataclasses are also JSON-serializable via `dataclasses.asdict()` for Phase 2 API.

### 2.2 Public API

```python
def validate(
    config: RAGPipelineConfig,
    corpus_tokens: int | None = None,
) -> ValidationResult:
    """
    Run all compatibility checks and cost estimates on a config.

    Args:
        config:        A validated RAGPipelineConfig instance.
        corpus_tokens: Optional corpus size for absolute cost estimates.
                       When None, cost estimates return per-million rates only.

    Returns:
        ValidationResult with .valid == True iff zero ERRORs.
    """
```

Single public function. Pure — no side effects, no I/O.

### 2.3 Internal Checker Architecture

```
validate()
  ├── _check_incompatible_pairs()     # iterates INCOMPATIBLE
  ├── _check_late_chunking()          # cross-field: chunking + embedding
  ├── _check_contextual_chunking()    # cross-field: context_model vs LLM
  ├── _check_flare_logprobs()         # FLARE × LLM provider
  ├── _check_hybrid_search_vectordb() # retrieval × vectordb
  ├── _check_sentence_window()        # sentence_window × framework
  ├── _check_multiple_advanced()      # at most one advanced technique
  ├── _check_reranker_top_n()         # reranker.top_n vs retrieval.top_k
  └── _estimate_costs()               # cost table × active components
```

Each checker receives `(config, issues)` or `(config, issues, costs, corpus_tokens)`.
Appends to lists in-place. No return values.

### 2.4 Dot-Path Resolution

The `_is_active(config, path) -> bool` helper resolves paths like:

```python
# Format: "section.subsection.type_value"
# Last segment = discriminator type OR enabled=True check

"generation.llm.anthropic"          → config.generation.llm.type == "anthropic"
"generation.advanced.flare"         → (config.generation.advanced is not None
                                       and config.generation.advanced.flare is not None
                                       and config.generation.advanced.flare.enabled)
"indexing.chunking.late"            → config.indexing.chunking.type == "late"
"retrieval.hybrid_rrf"              → config.retrieval.type == "hybrid_rrf"
"framework.langchain"               → config.framework == "langchain"
"indexing.vector_db.chromadb"       → config.indexing.vector_db.type == "chromadb"
"pre_retrieval.hyde.enabled"        → (config.pre_retrieval.hyde is not None
                                       and config.pre_retrieval.hyde.enabled)
```

**Critical**: `test_compatibility.py` has a meta-test that validates every dot-path
in `INCOMPATIBLE` and `WARNINGS` against actual config field names. This test fails
if `config.py` field names drift. Catches silent breakage.

### 2.5 Cost Estimator

```python
_COST_TABLE: dict[str, tuple[str, float]] = {
    # key → (description, USD per 1M tokens)
    "contextual_chunking": ("Contextual chunk annotation (with prompt caching)", 1.02),
    "proposition_chunking": ("Proposition extraction via gpt-4o-mini", 2.50),
}
```

- Iterates `_COST_TABLE`, checks if component is active in config.
- If `corpus_tokens` provided: `estimated_total = rate * corpus_tokens / 1_000_000`.
- All output includes disclaimer: "Estimate based on published pricing. Actual costs vary."
- Agentic RAG is NOT in the table (per-query, not per-corpus) — it goes into `WARNINGS` only.

### 2.6 Edge Cases to Handle

| Edge case | Behavior |
|---|---|
| FLARE enabled + Ollama LLM | WARNING (not ERROR) — some Ollama models expose logprobs, integration is unreliable |
| Contextual chunking + Ollama context_model | WARNING about throughput, not an ERROR |
| Late chunking + Jina v3 + `late_chunking=False` | ERROR with specific message about the flag |
| Late chunking + Jina v2 | ERROR with specific message about model version |
| reranker.top_n=5, retrieval.top_k=3 | WARNING: top_n > top_k, reranker will see fewer docs than expected |
| Multiple advanced techniques all disabled | No warning (disabled features don't count) |
| evaluation=None | No cost warning emitted |
| corpus_tokens=0 | estimated_total = 0.0 (valid edge case, not an error) |

---

## 3. generator.py

### 3.1 Design Principle: Pure Function, No Side Effects

```python
def generate(config: RAGPipelineConfig) -> GeneratorResult:
    """Generate pipeline code. Pure — never writes to disk."""
```

The CLI (Phase 1d) writes files. The API (Phase 2) returns JSON.
The generator knows nothing about either. This keeps it testable
without filesystem mocking and truly interface-agnostic.

### 3.2 Output Types

```python
@dataclass
class GeneratedFile:
    path:      str   # relative path, e.g. "pipeline.py"
    content:   str
    is_python: bool  # True → ast.parse() is run

@dataclass
class GeneratorResult:
    files:             dict[str, str]        # path → content (convenience)
    generated_files:   list[GeneratedFile]  # full detail
    validation_passed: bool                  # all ast.parse() checks passed
    errors:            list[str]            # ast.parse error messages
    config_yaml:       str                  # input config serialized to YAML
```

### 3.3 Template Loader

```python
class TemplateLoader:
    def __init__(self, template_dir: Path | None = None):
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir or _DEFAULT_TEMPLATE_DIR)),
            undefined=jinja2.StrictUndefined,  # fail fast — no silent empty strings
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_stage(self, category: str, type_name: str, ctx: dict) -> str:
        return self._env.get_template(f"stages/{category}/{type_name}.py.j2").render(ctx)

    def render_entrypoint(self, framework: str, name: str, ctx: dict) -> str:
        return self._env.get_template(f"entrypoints/{framework}/{name}.py.j2").render(ctx)

    def render_common(self, name: str, ctx: dict) -> str:
        return self._env.get_template(f"entrypoints/common/{name}.j2").render(ctx)
```

`StrictUndefined` is non-negotiable: a template referencing an undefined variable
must fail loudly, not silently render empty strings that produce broken Python.

### 3.4 Stage Rendering Order

Follows data flow (each stage depends on the one above):

```
1. chunking         ← chunking config
2. embedding        ← embedding config + (late_chunking flag from chunking)
3. vectordb         ← vectordb config
4. retrieval        ← retrieval config + vectordb type
5. reranker         ← post_retrieval.reranker (skip if None)
6. llm              ← generation.llm config
7. ingestion.py     ← composes: chunking + embedding + vectordb
8. pipeline.py      ← composes: all stages + pre/post_retrieval + advanced gen
9. pyproject.toml   ← get_dependencies(config) from versions.py
10. .env.example    ← _collect_required_env_vars(config)
11. README.md       ← config summary
12. Dockerfile      ← base image + deps (+ GPU flag if bge_m3 / cross_encoder)
13. docker-compose  ← ONLY if vectordb needs external service (qdrant/milvus/weaviate/pgvector)
```

### 3.5 Context Variables Per Template

```python
# Available to ALL templates
base_ctx = {
    "config":        config,               # full RAGPipelineConfig
    "framework":     config.framework,     # "langchain" | "llamaindex"
    "pipeline_name": config.name,
    "dependencies":  get_dependencies(config),
    "python_version": "3.11",
}

# Per-stage additions
chunking_ctx  = {**base_ctx, "chunking":   config.indexing.chunking}
embedding_ctx = {**base_ctx, "embedding":  config.indexing.embedding,
                              "is_late_chunking": config.indexing.chunking.type == "late"}
vectordb_ctx  = {**base_ctx, "vector_db":  config.indexing.vector_db}
retrieval_ctx = {**base_ctx, "retrieval":  config.retrieval,
                              "vector_db_type": config.indexing.vector_db.type}
reranker_ctx  = {**base_ctx, "reranker":   config.post_retrieval.reranker}
llm_ctx       = {**base_ctx, "llm":        config.generation.llm,
                              "prompt_template": config.generation.prompt_template,
                              "advanced":  config.generation.advanced}

# Entrypoints receive all rendered stage strings
entrypoint_ctx = {
    **base_ctx,
    "stages":       rendered_stages,   # dict[str, str]
    "pre_retrieval":  config.pre_retrieval,
    "post_retrieval": config.post_retrieval,
    "generation":     config.generation,
    "ingestion":      config.ingestion,
}

# Common templates
env_ctx = {**base_ctx, "env_vars": _collect_required_env_vars(config)}
```

### 3.6 Error Handling: Three Levels

```python
class GeneratorError(Exception):
    """Template or engine bug — not a user config error."""
    pass
```

| Error type | Cause | Handling |
|---|---|---|
| `jinja2.TemplateNotFound` | Missing template file | Wrap in `GeneratorError`. Return result with `validation_passed=False`. |
| `jinja2.UndefinedError` | Template references undefined var | Same — template bug, not user error. |
| `SyntaxError` from `ast.parse()` | Template rendered invalid Python | Append to `result.errors` with filename, line, context. |

`generate()` never raises. Always returns `GeneratorResult`. Caller handles errors.

### 3.7 ast.parse() Integration

```python
def _validate_python(code: str, filename: str) -> list[str]:
    try:
        ast.parse(code, filename=filename)
        return []
    except SyntaxError as e:
        return [
            f"{filename}:{e.lineno}: {e.msg}\n"
            f"  This is a bug in ragfactory templates. Please report it.\n"
            f"  Context: {e.text!r}"
        ]
```

Called on every `.py` file. NOT called on `.toml`, `.yml`, `.md`, `Dockerfile`.

### 3.8 Generated Files Summary

| File | Always? | ast.parse? | Notes |
|---|---|---|---|
| `pipeline.py` | Yes | Yes | Main RAG chain |
| `ingestion.py` | Yes | Yes | Document loading + chunking + indexing |
| `pyproject.toml` | Yes | No | From versions.get_dependencies() |
| `.env.example` | Yes | No | Only vars needed for this config |
| `README.md` | Yes | No | Config summary + quickstart |
| `Dockerfile` | Yes | No | GPU base image if bge_m3/cross_encoder |
| `docker-compose.yml` | Conditional | No | Only if qdrant/milvus/weaviate/pgvector |

---

## 4. Test Strategy

### 4.1 New Test Files

```
tests/
  unit/
    test_config.py           ✅ (existing, 102 tests)
    test_compatibility.py    NEW — data integrity
    test_validator.py        NEW — validation logic
    test_generator.py        NEW — generator engine (stub templates)
  fixtures/
    __init__.py
    stub_templates/          Minimal .j2 files for generator tests
      stages/chunking/recursive.py.j2
      stages/chunking/fixed.py.j2
      stages/embedding/openai.py.j2
      stages/vectordb/chromadb.py.j2
      stages/vectordb/qdrant.py.j2
      stages/retrieval/dense.py.j2
      stages/retrieval/hybrid_rrf.py.j2
      stages/llm/openai.py.j2
      stages/llm/anthropic.py.j2
      entrypoints/langchain/pipeline.py.j2
      entrypoints/langchain/ingestion.py.j2
      entrypoints/llamaindex/pipeline.py.j2
      entrypoints/llamaindex/ingestion.py.j2
      entrypoints/common/pyproject.toml.j2
      entrypoints/common/.env.example.j2
      entrypoints/common/README.md.j2
      entrypoints/common/Dockerfile.j2
```

Stub templates produce valid Python with minimal content:
```jinja2
{# stages/chunking/recursive.py.j2 #}
# chunking: {{ chunking.type }}
CHUNK_SIZE = {{ chunking.chunk_size }}
CHUNK_OVERLAP = {{ chunking.chunk_overlap }}
```

### 4.2 test_compatibility.py — Coverage

1. Data integrity: all `IncompatiblePair` entries have non-empty fields.
2. No duplicate pairs: `(component_a, component_b)` tuples are unique.
3. **Meta-test**: every dot-path in `INCOMPATIBLE` resolves to a real config field.
4. `WARNINGS` cost rates are positive floats when not None.
5. All severity values are valid `Severity` enum members.
6. `CROSS_FIELD_RULES` rule_ids are unique.

### 4.3 test_validator.py — Coverage

#### Incompatibility ERRORs

| Test | Config | Expected |
|---|---|---|
| `test_flare_anthropic_error` | FLARE + Anthropic | ERROR `FLARE_NO_LOGPROBS` |
| `test_flare_openai_ok` | FLARE + OpenAI | no error |
| `test_flare_cohere_error` | FLARE + Cohere | ERROR |
| `test_flare_ollama_warning` | FLARE + Ollama | WARNING (not ERROR) |
| `test_late_chunking_openai_error` | late + openai embed | ERROR |
| `test_late_chunking_jina_v3_flag_true_ok` | late + jina-v3 + flag=True | no error |
| `test_late_chunking_jina_v2_error` | late + jina-v2 | ERROR |
| `test_late_chunking_jina_flag_false_error` | late + jina-v3 + flag=False | ERROR |
| `test_hybrid_rrf_chromadb_error` | hybrid_rrf + chromadb | ERROR |
| `test_hybrid_rrf_qdrant_ok` | hybrid_rrf + qdrant | no error |
| `test_hybrid_weighted_pinecone_error` | hybrid_weighted + pinecone | ERROR |
| `test_sentence_window_langchain_error` | sentence_window + langchain | ERROR |
| `test_sentence_window_llamaindex_ok` | sentence_window + llamaindex | no error |

#### Cross-field WARNINGs

| Test | Config | Expected |
|---|---|---|
| `test_contextual_chunking_ollama_warns` | contextual + ollama context_model | WARNING throughput |
| `test_multiple_advanced_warns` | CRAG + FLARE both enabled | WARNING |
| `test_multiple_advanced_one_ok` | only CRAG enabled | no warning |
| `test_reranker_top_n_exceeds_top_k` | reranker top_n=20, retrieval top_k=10 | WARNING |
| `test_reranker_top_n_below_top_k_ok` | reranker top_n=5, retrieval top_k=20 | no warning |

#### Cost estimation

| Test | Input | Expected |
|---|---|---|
| `test_contextual_cost_no_corpus` | contextual chunking, no corpus_tokens | `estimated_total=None`, rate=1.02 |
| `test_contextual_cost_with_corpus` | contextual, corpus_tokens=1_000_000 | `estimated_total≈1.02` |
| `test_no_cost_simple_pipeline` | recursive+openai+dense | empty costs list |

#### Structural

| Test | Expected |
|---|---|
| `test_valid_config_is_valid_true` | `.valid == True` |
| `test_error_config_is_valid_false` | `.valid == False` |
| `test_errors_property_filters` | `.errors` returns only ERRORs |
| `test_warnings_property_filters` | `.warnings` returns only WARNINGs |
| `test_wrong_input_type_raises` | `TypeError` for non-RAGPipelineConfig |

### 4.4 test_generator.py — Coverage

Uses `stub_template_dir` pytest fixture (tmp_path with stub templates).

| Test | Verifies |
|---|---|
| `test_generate_returns_generator_result` | Return type |
| `test_generated_python_passes_ast` | All `.py` files parse cleanly |
| `test_all_expected_files_present` | pipeline.py, ingestion.py, pyproject.toml, .env.example, README.md, Dockerfile |
| `test_docker_compose_only_when_needed` | Present for qdrant, absent for chromadb |
| `test_generate_is_pure_no_disk_writes` | No files created in CWD |
| `test_langchain_framework_routes_correctly` | Uses langchain/ entrypoint |
| `test_llamaindex_framework_routes_correctly` | Uses llamaindex/ entrypoint |
| `test_config_yaml_in_result` | `result.config_yaml` round-trips to equal config |
| `test_missing_template_returns_error` | `validation_passed=False`, descriptive error |
| `test_strict_undefined_catches_missing_var` | Template with `{{ undefined_var }}` → error |
| `test_reranker_stage_skipped_when_none` | No reranker template rendered if reranker=None |
| `test_syntax_error_in_generated_py` | Stub template producing bad Python → error in result |
| `test_all_optional_fields_none` | Minimal config does not crash generator |

---

## 5. Implementation Order

### Strict sequence — each step must pass before next starts

```
Step 1  →  ragfactory/core/compatibility.py
Step 2  →  tests/unit/test_compatibility.py        [run: pytest test_compatibility.py]
Step 3  →  ragfactory/core/validator.py
Step 4  →  tests/unit/test_validator.py             [run: pytest test_validator.py]
Step 5  →  tests/fixtures/ + stub templates
Step 6  →  ragfactory/core/generator.py
Step 7  →  tests/unit/test_generator.py             [run: pytest test_generator.py]
Step 8  →  ragfactory/core/__init__.py (update exports)
Step 9  →  Full suite: pytest tests/unit/ -v        [ALL must pass]
Step 10 →  git commit: "feat: Phase 1b — compatibility matrix + validator + generator core"
```

### Dependency graph

```
config.py (Phase 1a)
    └── compatibility.py      (no project imports)
            └── validator.py  (imports config.py + compatibility.py)
versions.py (Phase 1a)
    └── generator.py          (imports config.py + versions.py)
                              (does NOT import validator.py — validation is caller's job)
```

**generator.py does NOT call validator.py**. Validation is the CLI/API's responsibility.
The generator trusts that the caller has already validated the config. This keeps
concerns separated: validate() catches logical incompatibilities, generate() just renders.

---

## 6. Risk Register

### R1 — Dot-Path Resolution Fragility (HIGH)

**Risk**: String-based paths like `"generation.advanced.flare"` in `INCOMPATIBLE` break
silently when `config.py` field names change.

**Mitigation**: `test_compatibility.py` meta-test validates every dot-path in
`INCOMPATIBLE` and `WARNINGS` against the actual config schema at test time.
If a path breaks, the test suite catches it before it ships.

**Residual risk**: Medium — meta-test must be maintained when new config fields are added.

---

### R2 — Generator Without Real Templates (MEDIUM)

**Risk**: `generator.py` API assumes template variable contracts. If Phase 1c templates
expect different variables, Phase 1c will require generator changes.

**Mitigation**: The context variable specification in §3.5 IS the template contract.
Phase 1c template authors follow this spec. Stub templates in tests serve as
living documentation. Any deviation in Phase 1c is a Phase 1c bug, not a generator bug.

**Residual risk**: Low — contracts are explicit and tested.

---

### R3 — Hybrid Search Compatibility Is Nuanced (HIGH)

**Risk**: ChromaDB CAN support hybrid if BM25 is implemented externally.
Pinecone Serverless DID add sparse-dense. Marking these as hard ERRORs
may frustrate users with working configurations.

**Mitigation**: Start strict (ERROR). As Phase 1c templates are written,
if a template successfully handles a previously-blocked combination,
downgrade from ERROR to WARNING in `compatibility.py`. The data is in
one place — one-line change to fix.

**Residual risk**: Low — conservative errors are safer than false positives.

---

### R4 — StrictUndefined Raises on Optional Template Vars (MEDIUM)

**Risk**: Some template variables are optional (e.g., `reranker` when no reranker
is configured). `StrictUndefined` will raise if a template references `reranker.top_n`
without checking for None.

**Mitigation**: Two approaches (decide in Phase 1c):
- **A (preferred)**: Don't pass `reranker_ctx` to non-reranker templates.
  Skip the reranker stage entirely when `config.post_retrieval.reranker is None`.
- **B**: Use Jinja2's `{% if reranker is not none %}` guards in templates.

Approach A is cleaner — template never sees None, so StrictUndefined works naturally.

---

### R5 — Circular Import: validator.py ↔ config.py (LOW)

**Risk**: validator.py imports from config.py. If config.py ever imports from validator.py,
circular import crashes the module system.

**Mitigation**: Dependency graph (§5) is acyclic by design. config.py is the root.
Nothing in config.py imports from validator.py or generator.py. This must be enforced
as a project rule: `config.py` has no project-level imports except stdlib + pydantic.

---

### R6 — Cost Estimates Misinterpreted as Guarantees (LOW)

**Risk**: Users budget based on cost estimates and are surprised by real bills.

**Mitigation**: Every `CostEstimate` output includes explicit disclaimer text.
The CLI will render these with a ⚠ prefix and a note about pricing variability.
All estimates are documented as conservative (with caching). No claim of accuracy.

---

## Verification Gate

Before Phase 1b is marked complete:

```bash
# All three new test files must pass
pytest tests/unit/test_compatibility.py -v   # data integrity
pytest tests/unit/test_validator.py -v        # validation logic
pytest tests/unit/test_generator.py -v        # generator engine

# Full suite regression — Phase 1a must stay green
pytest tests/unit/ -v
# Expected: 102 (Phase 1a) + new tests = all passing

# Type check
python -m mypy ragfactory/core/compatibility.py ragfactory/core/validator.py ragfactory/core/generator.py --strict

# Lint
python -m ruff check ragfactory/core/compatibility.py ragfactory/core/validator.py ragfactory/core/generator.py
```
