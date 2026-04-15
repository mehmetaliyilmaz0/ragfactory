# Phase 1d — CLI Tool

> **Status**: Ready for implementation
> **Date**: 2026-04-10
> **Architect**: Principal AI Architect review (v2 — post false-positive analysis)
> **Assignee**: Sonnet (implementation)
> **Prerequisite**: Phase 1c remediation committed at `4721938` ✓

---

## Scope

4 files, 1 phase, no sub-agents needed (all files tightly coupled):

| File | Status | LOC estimate |
|------|--------|-------------|
| `ragfactory/cli/__init__.py` | ❌ does not exist | 3 |
| `ragfactory/cli/main.py` | ❌ does not exist | ~310 |
| `tests/integration/test_cli.py` | ❌ does not exist | ~270 |
| `tests/fixtures/quick_start.yaml` | ❌ does not exist | ~20 |

**Entry point already wired in `pyproject.toml`:**
```toml
[project.scripts]
ragfactory = "ragfactory.cli.main:app"
```

---

## Pre-Implementation Audit

### `GeneratorResult` — exact attributes
```python
result.files             # dict[str, str]  — filename → content (6 files)
result.config_yaml       # str             — serialised YAML of the config used
result.validation_passed # bool            — AST syntax check ONLY, NOT compat check
result.errors            # list[str]       — AST syntax errors only
```

> ⚠️ **Critical distinction**: `result.validation_passed` reflects whether the
> generated `.py` files pass `ast.parse()` — it has NO relation to compatibility
> validation. `chromadb + hybrid_rrf` produces `validation_passed=True` because
> the generated code has no syntax errors, even though it's semantically invalid.
> The CLI MUST call `validate()` separately for compatibility gating. Never use
> `validation_passed` as a substitute.

### `ValidationResult` — exact attributes
```python
result.valid      # bool
result.errors     # list[ValidationIssue]   — hard incompatibilities
result.warnings   # list[ValidationIssue]   — soft warnings (do not block)
result.infos      # list[ValidationIssue]   — informational
result.costs      # list
```

### `ValidationIssue` — exact attributes
```python
issue.code           # str  e.g. "INCOMPAT_HYBRID_RRF_CHROMADB"
issue.message        # str  human-readable explanation
issue.severity       # ValidationSeverity enum
issue.component_path # str  e.g. "retrieval.hybrid_rrf"
issue.suggestion     # str | None
```

### `RAGPipelineConfig` required fields
- `name: str`
- `indexing: IndexingConfig` → requires `embedding` + `vector_db`; `chunking` has default
- `generation: GenerationConfig` → requires `llm`
- `retrieval` is **top-level**, NOT inside `indexing`

All VectorDB configs have sensible field defaults. No required-without-default fields.

### Exact output file list (verified)
`generate()` always produces exactly these 6 files:
```
pipeline.py, ingestion.py, pyproject.toml, .env.example, README.md, Dockerfile
```
`config_yaml` is a **separate attribute** — NOT in `result.files`. Must be written explicitly.

### Compatibility rules affecting `init` auto-selection
All 13 incompatible pairs from `compatibility.py`. The ones relevant to `init` flags:
```
retrieval.hybrid_rrf     x  indexing.vector_db.chromadb
retrieval.hybrid_weighted x  indexing.vector_db.chromadb
retrieval.hybrid_rrf     x  indexing.vector_db.pinecone
retrieval.hybrid_weighted x  indexing.vector_db.pinecone
retrieval.sentence_window x  framework.langchain
indexing.chunking.late   x  indexing.embedding.{openai,cohere,voyage,gemini,bge_m3,nomic}
```
The `validate()` call in each command surfaces all of these with clear messages.
The auto-select logic (below) only pre-handles the hybrid×DB pairs as they
are the most common UX trap and cannot be communicated before config construction.

### LLM type discriminator mismatch
Config discriminator for Cohere LLM is `"cohere_llm"`, not `"cohere"`.
CLI flag `--llm cohere` must map to `{"type": "cohere_llm"}` via `_LLM_TYPE_MAP`.

---

## Commands

### 1. `ragfactory generate`
```
ragfactory generate --config pipeline.yaml [--output ./my-rag] [--force]
```

Behaviour:
1. Load + parse config from `--config` (`.yaml`/`.yml` → PyYAML, `.json` → json.loads,
   other extension → error: `"Unsupported config format '{ext}'. Use .yaml or .json"`).
2. Run `validate(config)`. Print all errors/warnings/infos via `_print_validation()`.
3. If `errors` present AND `--force` not set → `typer.Exit(1)`.
4. If `--force` set, or `errors` is empty → run `generate(config)`.
5. Write all `result.files` + `result.config_yaml` (as `config.yaml`) to `--output` dir.
6. Print file summary table via `_print_file_summary()`.
7. `typer.Exit(0)`.

Output dir default: `Path(config.name.replace("/", "-"))` (sanitize `/` only; spaces are valid).

### 2. `ragfactory validate`
```
ragfactory validate --config pipeline.yaml
```

Behaviour:
1. Load + parse config (same extension logic as `generate`).
2. Run `validate(config)`.
3. Print errors (red), warnings (yellow), infos (blue) via `_print_validation()`.
4. Final line: `✓ Valid` (green) if `result.valid`, else `✗ Invalid — N error(s)` (red).
5. `typer.Exit(0)` if `result.valid`, else `typer.Exit(1)`.

### 3. `ragfactory init`
```
ragfactory init --name my-pipeline [OPTIONS]
```

Flags:

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--name` | TEXT | (required) | Pipeline name |
| `--framework` | Choice | `langchain` | `langchain \| llamaindex` |
| `--chunking` | Choice | `recursive` | `fixed\|recursive\|semantic\|contextual\|late\|page_level\|sentence_window` |
| `--embedding` | Choice | `openai` | `openai\|cohere\|voyage\|gemini\|bge_m3\|nomic\|jina` |
| `--vector-db` | Choice | `chromadb` | `chromadb\|qdrant\|pinecone\|weaviate\|milvus\|pgvector` |
| `--retrieval` | Choice | `None` (auto) | `dense\|hybrid_rrf\|hybrid_weighted\|small_to_big\|sentence_window` |
| `--reranker` | Choice | `none` | `none\|cohere\|cross_encoder\|colbert\|flashrank` |
| `--llm` | Choice | `openai` | `openai\|anthropic\|cohere\|ollama` |
| `--output` | PATH | `./{name}` | Output directory for generated files |
| `--save-config` | PATH | None | Write config YAML to this path |

**`--retrieval` must be `Optional[str] = typer.Option(None)`** so `None` means
"not explicitly passed" — auto-select logic only fires when it's `None`.

**Retrieval auto-selection** (applied only when `--retrieval` not passed):
```python
_DENSE_ONLY_DBS = {"chromadb", "pinecone"}

def _resolve_retrieval(vector_db: str, retrieval: str | None) -> str:
    if retrieval is not None:
        return retrieval  # user's explicit choice, pass through
    return "dense" if vector_db in _DENSE_ONLY_DBS else "hybrid_rrf"
```

**LLM alias mapping:**
```python
_LLM_TYPE_MAP: dict[str, str] = {
    "openai":    "openai",
    "anthropic": "anthropic",
    "cohere":    "cohere_llm",   # discriminator mismatch — must map
    "ollama":    "ollama",
}
```

**`--save-config` + `--output` combined behaviour:**
- `--save-config` only (no `--output`): write YAML to that path, skip code generation, exit 0.
- `--output` only (no `--save-config`): generate code to output dir, config.yaml also written there.
- Both set: generate code to `--output` AND write YAML to `--save-config` path.
- Neither set: generate code to `./{name}`, config.yaml also written there.

Behaviour:
1. Build `RAGPipelineConfig` via `_build_config_from_flags()` (see below).
2. Run `validate(config)`. Print warnings (proceed) or errors (abort with `typer.Exit(1)`).
3. If `--save-config` set → write `config.model_dump_yaml()` to that path.
4. If not (`--save-config`-only mode) → exit 0.
5. Otherwise → run `generate(config)`, write files + `config.yaml` to `--output`, print summary.

### 4. `ragfactory options`
```
ragfactory options [--component COMPONENT] [--json]
```

Flags:
- `--component / -c`: one of `chunking | embedding | vectordb | retrieval | reranker | llm`
  (note: `vectordb` without underscore — user-facing convention)
- `--json`: emit machine-readable JSON

**All-components view** (no `--component`): single Rich Table with columns
`Component | Type | Description`, all 6 components interleaved, grouped by component
with a blank separator row between groups.

**Single-component view** (`--component embedding`): Rich Table with columns
`Type | Description` for that component only.

**JSON output schema** (for `--json`):
```json
{
  "chunking":  [{"type": "fixed",   "description": "Fixed-size token windows..."}, ...],
  "embedding": [{"type": "openai",  "description": "OpenAI text-embedding-3-small/large..."}, ...],
  "vectordb":  [...],
  "retrieval": [...],
  "reranker":  [...],
  "llm":       [...]
}
```
If `--component` + `--json`: emit just the array for that component (not wrapped in object).

Component registry (static):
```python
_COMPONENTS: dict[str, list[tuple[str, str]]] = {
    "chunking": [
        ("fixed",           "Fixed-size token windows. Simple baseline."),
        ("recursive",       "Hierarchical separators. Production default."),
        ("semantic",        "Embedding-based breakpoints. Handles topic drift."),
        ("contextual",      "Prepends LLM context to each chunk. Best recall (+49-67%)."),
        ("late",            "Jina Late Chunking. Token-level pooling after full encoding."),
        ("page_level",      "One chunk per PDF page. Good for structured docs."),
        ("sentence_window", "Fine-grained sentence retrieval with window expansion."),
    ],
    "embedding": [
        ("openai",  "OpenAI text-embedding-3-small/large. Default 1536d."),
        ("cohere",  "Cohere embed-v3. 1024d. Multilingual."),
        ("voyage",  "Voyage AI. Best retrieval benchmarks (MTEB 2024)."),
        ("gemini",  "Google text-embedding-004. 768d."),
        ("bge_m3",  "BAAI/BGE-M3. Self-hosted. Dense+sparse+colbert."),
        ("nomic",   "Nomic embed-text. Self-hosted or API. 768d."),
        ("jina",    "Jina embeddings-v3. 1024d. Long context (8192 tokens)."),
    ],
    "vectordb": [
        ("chromadb", "Embedded, in-process. Prototyping only."),
        ("qdrant",   "Rust. Production default. 8500-12000 QPS. Best metadata filtering."),
        ("pinecone", "Serverless. Zero-ops. Up to 1.4B vectors."),
        ("weaviate", "Modular. Native hybrid. Multi-tenancy."),
        ("milvus",   "Distributed. GPU-accelerated (CAGRA). Billion-scale."),
        ("pgvector", "PostgreSQL extension. Best for existing Postgres stacks."),
    ],
    "retrieval": [
        ("dense",           "Pure vector similarity. Fast baseline."),
        ("hybrid_rrf",      "BM25 + dense via Reciprocal Rank Fusion. +15-30% recall."),
        ("hybrid_weighted", "BM25 + dense with tunable alpha weight."),
        ("small_to_big",    "Retrieve child chunks, return parent context."),
        ("sentence_window", "Retrieve sentence, expand to window."),
    ],
    "reranker": [
        ("cohere",        "Cohere Rerank API. Fast, no GPU needed."),
        ("cross_encoder", "Cross-encoder. Best quality, GPU recommended."),
        ("colbert",       "ColBERT (RAGatouille). Token-level interaction."),
        ("flashrank",     "FlashRank. Fastest local reranker. CPU-friendly."),
    ],
    "llm": [
        ("openai",     "GPT-4o / GPT-4o-mini. Default gpt-4o-mini."),
        ("anthropic",  "Claude Sonnet/Opus/Haiku. Default claude-sonnet-4-6."),
        ("cohere",     "Command-R+. Grounding optimised."),
        ("ollama",     "Local inference. Zero API cost. Requires Ollama."),
    ],
}
```

---

## Internal Helpers (in `main.py`)

```python
# ── Config loading ──────────────────────────────────────────────────────────
def _load_config(path: Path) -> RAGPipelineConfig:
    """Load YAML or JSON config. Prints clean error + typer.Exit(1) on failure.
    Supported: .yaml, .yml, .json. Other extensions → explicit error message."""

# ── Config construction (init command only) ─────────────────────────────────
def _build_config_from_flags(
    name: str,
    framework: str,
    chunking: str,
    embedding: str,
    vector_db: str,
    retrieval: str | None,
    reranker: str | None,
    llm: str,
) -> RAGPipelineConfig:
    """Construct RAGPipelineConfig from init CLI flags.

    NOTE: retrieval is a TOP-LEVEL field, NOT inside indexing.
    NOTE: reranker = None or "none" → post_retrieval remains default (no reranker).
    """
    resolved_retrieval = _resolve_retrieval(vector_db, retrieval)
    post_retrieval_dict: dict | None = (
        {"reranker": {"type": reranker}} if reranker and reranker != "none" else None
    )
    return RAGPipelineConfig(
        name=name,
        framework=framework,
        indexing={
            "chunking":  {"type": chunking},
            "embedding": {"type": embedding},
            "vector_db": {"type": vector_db},
        },
        retrieval={"type": resolved_retrieval},           # TOP-LEVEL
        post_retrieval=post_retrieval_dict,
        generation={"llm": {"type": _LLM_TYPE_MAP[llm]}},
    )

# ── Output writing ──────────────────────────────────────────────────────────
def _write_output(
    files: dict[str, str],
    config_yaml: str,
    output_dir: Path,
) -> list[Path]:
    """Create output_dir, write result.files + config.yaml, return written paths."""

# ── Display ─────────────────────────────────────────────────────────────────
def _print_validation(result: ValidationResult, console: Console) -> None:
    """Print errors (red), warnings (yellow), infos (blue) via Rich."""

def _print_file_summary(written: list[Path], console: Console) -> None:
    """Print Rich Table: filename | bytes | ✓"""
```

---

## `ragfactory --version`

Add version callback to the root `app`:
```python
def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ragfactory {ragfactory.__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(None, "--version", callback=_version_callback, is_eager=True),
) -> None:
    """Generate production-ready RAG pipelines from a single config."""
```

---

## `tests/fixtures/quick_start.yaml`

Create this file for use in manual verification and as a reference config:
```yaml
# ragfactory quick-start config — recursive chunking, OpenAI, ChromaDB, dense retrieval
name: quick-start
framework: langchain
indexing:
  chunking:
    type: recursive
    chunk_size: 512
    chunk_overlap: 50
  embedding:
    type: openai
  vector_db:
    type: chromadb
retrieval:
  type: dense
generation:
  llm:
    type: openai
```

---

## Integration Test Plan (`tests/integration/test_cli.py`)

Use `typer.testing.CliRunner` — no subprocess, no mocking, real filesystem, real core.

### Fixture design
```python
from typer.testing import CliRunner
from ragfactory.cli.main import app
import textwrap, ast, json

runner = CliRunner()

@pytest.fixture()
def valid_config(tmp_path: Path) -> Path:
    """chromadb + dense — always valid."""
    f = tmp_path / "valid.yaml"
    f.write_text(textwrap.dedent("""
        name: test-pipeline
        framework: langchain
        indexing:
          chunking: {type: recursive}
          embedding: {type: openai}
          vector_db: {type: chromadb}
        retrieval: {type: dense}
        generation:
          llm: {type: openai}
    """))
    return f

@pytest.fixture()
def incompatible_config(tmp_path: Path) -> Path:
    """chromadb + hybrid_rrf — triggers INCOMPAT_HYBRID_RRF_CHROMADB."""
    f = tmp_path / "bad.yaml"
    f.write_text(textwrap.dedent("""
        name: bad-pipeline
        framework: langchain
        indexing:
          chunking: {type: recursive}
          embedding: {type: openai}
          vector_db: {type: chromadb}
        retrieval: {type: hybrid_rrf}
        generation:
          llm: {type: openai}
    """))
    return f

@pytest.fixture()
def warning_only_config(tmp_path: Path) -> Path:
    """chromadb — triggers WARN_CHROMADB warning but valid=True."""
    f = tmp_path / "warn.yaml"
    f.write_text(textwrap.dedent("""
        name: warn-pipeline
        framework: langchain
        indexing:
          chunking: {type: recursive}
          embedding: {type: openai}
          vector_db: {type: chromadb}
        retrieval: {type: dense}
        generation:
          llm: {type: openai}
    """))
    return f
```

> **Rule**: Every test that invokes a command writing files to the default output
> path (i.e. without explicit `--output`) MUST use `monkeypatch.chdir(tmp_path)`
> before invoking the runner. Tests with explicit absolute `--output` paths do not
> need chdir.
>
> **Rule**: Every test that asserts `exit_code == 0` MUST also assert
> `result.exception is None`. CliRunner catches exceptions silently and maps them
> to exit_code=1, hiding the real error.

### Test cases (16 tests)

| Test | Command | Assertions |
|------|---------|------------|
| `test_generate_creates_all_files` | `generate --config valid.yaml --output {tmp}/out` | All 6 generated files + `config.yaml` exist (7 total) |
| `test_generate_python_files_parse` | same | `ast.parse()` passes for `pipeline.py` and `ingestion.py` |
| `test_generate_config_yaml_written` | same | `config.yaml` exists in output dir and is valid YAML |
| `test_generate_default_output_dir` | `generate --config valid.yaml` (no --output) + `monkeypatch.chdir(tmp)` | Dir `{tmp}/test-pipeline` created |
| `test_generate_invalid_config_exits_1` | `generate --config bad.yaml --output {tmp}/out` | `exit_code == 1`, `result.exception is None` |
| `test_generate_force_skips_validation` | `generate --config bad.yaml --force --output {tmp}/out` | `exit_code == 0`, files created, `result.exception is None` |
| `test_generate_missing_file_exits_1` | `generate --config nonexistent.yaml` | `exit_code == 1`, output contains "not found" |
| `test_validate_valid_exits_0` | `validate --config valid.yaml` | `exit_code == 0`, `result.exception is None` |
| `test_validate_invalid_exits_1` | `validate --config bad.yaml` | `exit_code == 1`, output contains `"INCOMPAT_HYBRID_RRF_CHROMADB"` |
| `test_validate_shows_warnings` | `validate --config warn.yaml` | `exit_code == 0` (warnings don't block), output contains `"WARN_CHROMADB"` |
| `test_options_all_components` | `options` | output contains `"chunking"`, `"embedding"`, `"vectordb"` |
| `test_options_filter_component` | `options --component embedding` | output contains `"openai"`, `"voyage"`, does NOT contain `"chromadb"` |
| `test_options_json_output` | `options --json` | output is valid JSON, top-level keys include `"chunking"`, `"llm"` |
| `test_init_creates_output` | `init --name foo --vector-db chromadb --llm openai --output {tmp}/foo` | `{tmp}/foo/pipeline.py` exists, `exit_code == 0`, `result.exception is None` |
| `test_init_save_config_only` | `init --name foo --save-config {tmp}/foo.yaml` | `foo.yaml` written and parseable, no `pipeline.py` in `tmp_path` |
| `test_init_auto_retrieval_chromadb` | `init --name foo --vector-db chromadb --output {tmp}/foo` | `"dense"` in `{tmp}/foo/pipeline.py`, `"hybrid_rrf"` not in `pipeline.py` |
| `test_init_auto_retrieval_qdrant` | `init --name foo --vector-db qdrant --output {tmp}/foo` | `"hybrid_rrf"` or `"HybridRRF"` in `{tmp}/foo/pipeline.py` |
| `test_help_exits_0` | `--help` | `exit_code == 0`, output contains `"generate"`, `"validate"`, `"init"`, `"options"` |
| `test_version_flag` | `--version` | `exit_code == 0`, output contains `"ragfactory"` and a version number |

(19 tests total — replace "16 tests" heading with "19 tests minimum")

---

## Implementation Order

Implement in this exact sequence. Write tests for each command before moving to the next.

1. **`cli/__init__.py`** — `from .main import app` (re-export for clean imports in tests)
2. **Imports + constants** — `app = typer.Typer(...)`, `_COMPONENTS`, `_LLM_TYPE_MAP`, `_DENSE_ONLY_DBS`
3. **`--version` callback** — root `app.callback()`
4. **`_load_config()`** — covers `FileNotFoundError`, `YAMLError`, `json.JSONDecodeError`, `ValidationError`, unknown extension
5. **`_resolve_retrieval()`** — 3 lines, pure function
6. **`_build_config_from_flags()`** — exact dict structure as shown above
7. **`_write_output()`** — mkdir + write loop + writes `config.yaml` from `config_yaml` param
8. **`_print_validation()`** — Rich errors/warnings/infos
9. **`_print_file_summary()`** — Rich Table
10. **`options` command** + tests
11. **`validate` command** + tests
12. **`generate` command** + tests
13. **`init` command** + tests
14. **`tests/fixtures/quick_start.yaml`**

---

## Verification Checklist

All paths Windows-compatible. Run from repo root.

```powershell
# 0. Editable install (if not already done)
pip install -e .

# 1. CLI entry point works
ragfactory --help
ragfactory --version

# 2. Options command
ragfactory options
ragfactory options --component embedding
ragfactory options --json

# 3. Validate with fixture
ragfactory validate --config tests/fixtures/quick_start.yaml

# 4. Generate from fixture
ragfactory generate --config tests/fixtures/quick_start.yaml --output test-output

# 5. Generated Python is syntactically valid
python -c "import ast; ast.parse(open('test-output/pipeline.py').read()); print('pipeline.py OK')"
python -c "import ast; ast.parse(open('test-output/ingestion.py').read()); print('ingestion.py OK')"
python -c "import yaml; yaml.safe_load(open('test-output/config.yaml').read()); print('config.yaml OK')"

# 6. Init works — 2 MVP combinations
ragfactory init --name quick-start --vector-db chromadb --llm openai --output test-init-1
ragfactory init --name prod-standard --vector-db qdrant --embedding voyage --llm anthropic --output test-init-2

# 7. Init auto-retrieval check
python -c "
content = open('test-init-1/pipeline.py').read()
assert 'dense' in content.lower() or 'Dense' in content, 'chromadb should use dense'
print('auto-retrieval chromadb: OK')
content2 = open('test-init-2/pipeline.py').read()
assert 'hybrid' in content2.lower() or 'Hybrid' in content2, 'qdrant should use hybrid_rrf'
print('auto-retrieval qdrant: OK')
"

# 8. Integration tests
pytest tests/integration/test_cli.py -v

# 9. Full suite — no regressions
pytest

# 10. Cleanup
Remove-Item -Recurse -Force test-output, test-init-1, test-init-2
```

---

## What NOT to build in Phase 1d

- ❌ Interactive TUI wizard (prompt_toolkit, questionary) — future phase
- ❌ `--watch` mode / hot reload
- ❌ Shell completion scripts
- ❌ `ragfactory upgrade` command
- ❌ Plugin system

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CliRunner swallows exceptions → exit_code=1 with no traceback | High | `assert result.exception is None` on every exit_code==0 test |
| `--retrieval` defaulting: must be `Optional[str] = typer.Option(None)` not `str = typer.Option("dense")` | High | Auto-select fires only when `retrieval is None` — cannot detect explicit pass with a string default |
| `generate()` overwrites existing files silently | Medium | Warn (not error) if `output_dir` is non-empty before writing |
| `config.name` contains `/` → `Path(config.name)` creates nested dirs | Low | `config.name.replace("/", "-")` in output dir derivation |
| CWD-sensitive tests pollute repo root | Medium | `monkeypatch.chdir(tmp_path)` mandatory for any test using default output path |
| Rich strips color in non-TTY CliRunner context | Low | Assert on text content only, never on ANSI escape codes |
