# Phase 1b Fixes — Implementation Plan

> **Authored by:** Principal AI Architect + Lead AI Researcher  
> **Scope:** Fix all findings from the Phase 1b code review (post false-positive correction)  
> **Constraint:** Max 5 files per phase · All tests green after every phase · No public API changes

---

## 1. Executive Summary

This plan remediates 12 findings across `validator.py`, `generator.py`, tests, and `pyproject.toml`.
The work is sequenced so that:
- (a) Dead code is deleted before any structural refactor touches the same region (CLAUDE.md Step 0 Rule)
- (b) The new shared helper `_providers.py` exists before either consumer imports from it
- (c) Each phase ends with the full test suite green and the touched file count ≤ 5
- No public API signatures (`validate`, `generate`) change across any phase
- Test count starts at 227 and only grows

---

## 2. Dependency Graph

```
Phase 0 — Dead code removal (Step 0 Rule)
  └─ M2: delete _check_hybrid_search_vectordb + _check_sentence_window_framework
         must land before any other edit to validator.py to keep diffs clean

Phase 1 — Shared helper (prerequisite for M6)
  └─ CREATE ragfactory/core/_providers.py
         ↓ enables
Phase 2 — validator.py: M3 fix + M6 consumer + M1 tests
  ├─ M3: fix local-model heuristic (uses _providers)
  ├─ M6 (validator side): import infer_context_model_provider
  └─ M1: add TestIsActive unit tests

Phase 3 — generator.py: M4, M6 consumer, M7, M9, C1, M5, m12
  ├─ M6 (generator side): import infer_context_model_provider
  ├─ M4: extract _render helper
  ├─ M7: delete redundant _add("openai") branch
  ├─ M9: compute fw once in generate(), pass into _render_stages
  ├─ C1: comment dead .value branch
  ├─ M5: comment dual-access contract
  └─ m12: importlib.resources template dir

Phase 4 — GeneratorResult property + test hygiene (M8, m4, C2)
  ├─ M8: GeneratorResult.files → @property
  ├─ m4: monkeypatch.chdir
  └─ C2: parametrized stub-template meta-test

Phase 5 — Packaging
  └─ m12 (pyproject.toml side): hatch package-data for templates
```

**Ordering rationale:** `_providers.py` must exist before Phases 2 and 3 can import from it. M2 dead-code deletion is Step 0 per CLAUDE.md. M8 touches `GeneratorResult` shape — deferred to its own phase so generator refactors in Phase 3 don't collide with test-surface changes.

---

## 3. Phases

### Phase 0 — Dead code removal (Step 0)

**Goal:** Remove the two empty extension-point functions before any structural refactor.

**Files touched (1):**
- `ragfactory/core/validator.py`

**Changes:**
- Delete `_check_hybrid_search_vectordb` (lines 366–377) entirely, including its docstring and trailing comment.
- Delete `_check_sentence_window_framework` (lines 380–388) entirely.
- In `validate()` (lines 553–562), remove the two call lines:
  - `_check_hybrid_search_vectordb(config, issues)` (line 557)
  - `_check_sentence_window_framework(config, issues)` (line 558)
- Leave ordering of remaining checkers unchanged.

**Verification:** `python -m pytest tests/ -q`  
**Expected test count:** 227 (unchanged — these functions had no direct tests).

---

### Phase 1 — Create shared `_providers.py`

**Goal:** Provide a single source of truth for context-model → provider inference, consumed by both validator and generator in later phases.

**Files touched (1):**
- `ragfactory/core/_providers.py` (NEW)

**Module contents:**

1. Module docstring: internal helper, not part of public API, pure data + pure function, zero project imports.

2. Cloud provider prefix table:
```python
# Each entry: (prefixes_tuple, provider_key)
# Ordered from most-specific to least-specific.
_CLOUD_PROVIDER_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gpt-", "o1-", "o1", "text-embedding-", "text-"), "openai"),
    (("claude-",),                                        "anthropic"),
    (("command",),                                        "cohere_llm"),
    (("gemini-",),                                        "gemini"),
    (("mistral-", "mistral"),                             "mistral"),
    (("bedrock/",),                                       "bedrock"),
)
```

3. Public function:
```python
def infer_context_model_provider(model_name: str) -> str | None:
    """
    Return the provider key for a known cloud model name prefix, or None
    if the model is unrecognised (could be local or an unsupported provider).
    """
    for prefixes, provider in _CLOUD_PROVIDER_PREFIXES:
        if any(model_name.startswith(p) for p in prefixes):
            return provider
    return None
```

4. Local-model helper:
```python
def is_probably_local_model(model_name: str) -> bool:
    """
    Heuristic: a model is 'probably local' if it has no known cloud prefix
    and contains no path separator (HuggingFace: 'org/model') or port
    separator (Ollama: 'model:tag').
    """
    return (
        infer_context_model_provider(model_name) is None
        and "/" not in model_name
        and ":" not in model_name
    )
```

5. Env-var lookup table (exported for both validator and generator):
```python
PROVIDER_ENV_VAR: dict[str, str] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "cohere_llm": "COHERE_API_KEY",
    "gemini":     "GOOGLE_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "bedrock":    "AWS_BEARER_TOKEN_BEDROCK",
}
```

**No imports from** `ragfactory.core.*` — keeps dependency arrow one-way to avoid circular imports.

**Verification:** `python -m pytest tests/ -q`  
**Expected test count:** 227 (new module has no tests yet; consumed in Phase 2).

---

### Phase 2 — validator.py: adopt `_providers`, fix M3, add TestIsActive

**Goal:** Fix cloud-model misclassification (M3), deduplicate provider inference (M6 consumer), add direct unit tests for `_is_active` (M1).

**Files touched (2):**
- `ragfactory/core/validator.py`
- `tests/unit/test_validator.py`

#### Changes to `validator.py`

1. **Add import** near line 30:
```python
from ragfactory.core._providers import (
    PROVIDER_ENV_VAR,
    infer_context_model_provider,
    is_probably_local_model,
)
```

2. **Rewrite `_check_contextual_chunking`** (lines 259–334). Replace the entire body with:
```python
def _check_contextual_chunking(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    if config.indexing.chunking.type != "contextual":
        return

    chunking = config.indexing.chunking
    context_model: str = chunking.context_model  # type: ignore[union-attr]

    context_provider = infer_context_model_provider(context_model)

    if context_provider is None:
        # Unrecognised provider. Check if it looks like a local model.
        if is_probably_local_model(context_model):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL",
                message=(
                    f"context_model='{context_model}' appears to be a local Ollama model. "
                    "Contextual chunking makes one LLM call per chunk — at local inference "
                    "speeds this is ~100x slower than Claude Haiku (API). "
                    "A 10K-chunk corpus may take hours. Consider using an API model."
                ),
                component_path="indexing.chunking.context_model",
                suggestion="Use 'claude-3-haiku-20240307' (cheap) or 'gpt-4o-mini' for context generation.",
            ))
        # Unknown cloud-like model (has / or :) — emit nothing; user knows what they're doing.
        return

    # Known cloud provider: check if it requires an extra API key beyond the main LLM.
    llm_type = config.generation.llm.type
    llm_provider_map = {
        "openai":     "openai",
        "anthropic":  "anthropic",
        "cohere_llm": "cohere_llm",
        "ollama":     "ollama",
    }

    if context_provider != llm_provider_map.get(llm_type):
        env_var = PROVIDER_ENV_VAR.get(context_provider, f"{context_provider.upper()}_API_KEY")
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            code="CONTEXTUAL_CHUNKING_EXTRA_API_KEY",
            message=(
                f"context_model='{context_model}' uses the {context_provider} API, "
                f"but your pipeline LLM uses {llm_type}. "
                f"An extra API key is required: {env_var}."
            ),
            component_path="indexing.chunking.context_model",
            suggestion=f"Add {env_var} to your .env file.",
        ))
```

3. Do NOT touch any other function.

#### Changes to `tests/unit/test_validator.py`

**Add `class TestIsActive`** with direct imports and assertions:
```python
from ragfactory.core.validator import _is_active
```

Cover every if-branch in `_is_active` (validator.py:107–188). One positive + one negative per pattern:

| Path | Positive config | Negative config |
|---|---|---|
| `framework.langchain` | `framework=Framework.LANGCHAIN` | `framework=Framework.LLAMAINDEX` |
| `framework.llamaindex` | `framework=Framework.LLAMAINDEX` | `framework=Framework.LANGCHAIN` |
| `indexing.chunking.<type>` | one per ChunkingConfig literal | different type |
| `indexing.embedding.<type>` | one per EmbeddingConfig literal | different type |
| `indexing.vector_db.<type>` | one per VectorDBConfig literal | different type |
| `retrieval.<type>` | one per RetrievalConfig literal | different type |
| `pre_retrieval.hyde` | `hyde=HyDEConfig(enabled=True)` | `hyde=None`, `hyde=HyDEConfig(enabled=False)` |
| `pre_retrieval.query_rewriting` | enabled=True | None or disabled |
| `pre_retrieval.routing` | enabled=True | None or disabled |
| `post_retrieval.reranker.<type>` | one per RerankerConfig literal | `reranker=None` |
| `generation.llm.<type>` | one per LLMConfig literal | different type |
| `generation.advanced.flare` | enabled=True | `advanced=None`, `flare=None`, `enabled=False` |
| `generation.advanced.crag` | same pattern | — |
| `generation.advanced.agentic` | same pattern | — |
| `generation.advanced.crag.web_search_fallback` | crag enabled + web_search_fallback=True | web_search_fallback=False |
| `evaluation` | `evaluation=EvaluationConfig()` | `evaluation=None` |
| `ingestion.parser.<value>` | one per ParserType literal | different value |
| `does.not.exist` (unknown) | — | always False |

**Add `class TestContextualChunkingCloudModels`** (M3 regression):
- `gemini-1.5-flash` as `context_model`, LLM=OpenAI → assert NO `CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL` issue; assert HAS `CONTEXTUAL_CHUNKING_EXTRA_API_KEY` with `GOOGLE_API_KEY`.
- `command-r-plus` as `context_model`, LLM=OpenAI → assert HAS `CONTEXTUAL_CHUNKING_EXTRA_API_KEY` with `COHERE_API_KEY`.
- `mistral-large` as `context_model`, LLM=OpenAI → assert HAS `CONTEXTUAL_CHUNKING_EXTRA_API_KEY` with `MISTRAL_API_KEY`.
- `llama3.2` as `context_model`, LLM=OpenAI → assert HAS `CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL` (regression: old behavior for real local models preserved).
- `gpt-4o-mini` as `context_model`, LLM=OpenAI → assert NO issues of either code (same provider).
- `claude-3-haiku-20240307` as `context_model`, LLM=OpenAI → assert HAS `CONTEXTUAL_CHUNKING_EXTRA_API_KEY` with `ANTHROPIC_API_KEY`.

**Verification:** `python -m pytest tests/ -q`  
**Expected test count:** ≥ 270 (227 + ~40 TestIsActive + 6 TestContextualChunkingCloudModels).

---

### Phase 3 — generator.py: M4, M6 consumer, M7, M9, C1, M5, m12

**Goal:** Structural cleanups to generator.py. Public `generate()` signature unchanged.

**Files touched (1):**
- `ragfactory/core/generator.py`

**Changes top-to-bottom:**

1. **m12** — Replace `_DEFAULT_TEMPLATE_DIR` (line 44):
```python
from importlib.resources import files as _pkg_files

_DEFAULT_TEMPLATE_DIR = Path(str(_pkg_files("ragfactory") / "templates"))
# NOTE: Path(str(...)) converts the Traversable returned by importlib.resources
# to a real Path, which jinja2.FileSystemLoader requires. This also works in
# zip-wheel installs where Path(__file__) would fail.
```
Keep `from pathlib import Path` — still used for type hints.

2. **M6** — Add import:
```python
from ragfactory.core._providers import infer_context_model_provider
```

3. **M4** — Add private `_render` method to `TemplateLoader`, then slim down all three public render methods:
```python
def _render(self, template_path: str, ctx: dict) -> str:
    try:
        return self._env.get_template(template_path).render(ctx)
    except jinja2.TemplateNotFound:
        raise GeneratorError(
            f"Template not found: {template_path}\n"
            "This is a bug in ragfactory. Please report it at "
            "https://github.com/mehmetaliyilmaz0/ragfactory/issues"
        ) from None
    except jinja2.UndefinedError as e:
        raise GeneratorError(
            f"Template variable error in {template_path}: {e}\n"
            "This is a bug in ragfactory templates."
        ) from e

def render_stage(self, category: str, type_name: str, ctx: dict) -> str:
    """Render stages/<category>/<type_name>.py.j2"""
    return self._render(f"stages/{category}/{type_name}.py.j2", ctx)

def render_entrypoint(self, framework: str, name: str, ctx: dict) -> str:
    """Render entrypoints/<framework>/<name>.py.j2"""
    return self._render(f"entrypoints/{framework}/{name}.py.j2", ctx)

def render_common(self, name: str, ctx: dict) -> str:
    """Render entrypoints/common/<name>.j2"""
    return self._render(f"entrypoints/common/{name}.j2", ctx)
```
Keep all docstrings. Error messages now uniformly include "Please report it" — this is strictly richer than before and existing `pytest.raises(GeneratorError, match="Template not found")` assertions still match.

4. **M7** — In `_collect_required_env_vars`, delete lines 216–217:
```python
# DELETED — _add(config.generation.llm.type) on the line above already
# adds OPENAI_API_KEY when llm.type=="openai", and `seen` deduplicates.
```

5. **M6 consumer** — In `_collect_required_env_vars`, replace the contextual-chunking block (lines 220–225):
```python
# Contextual chunking may need extra API key for the context model
if config.indexing.chunking.type == "contextual":
    ctx_model: str = config.indexing.chunking.context_model  # type: ignore[union-attr]
    provider = infer_context_model_provider(ctx_model)
    if provider is not None:
        _add(provider)
```
This now correctly handles Gemini (`GOOGLE_API_KEY`), Mistral (`MISTRAL_API_KEY`), Cohere (`COHERE_API_KEY`) — matching validator behaviour.

6. **C1** — Replace `_framework_str` body with a commented version:
```python
def _framework_str(config: RAGPipelineConfig) -> str:
    """Return framework as a plain str path segment for template path construction."""
    # use_enum_values=True on StrictModel guarantees config.framework is already
    # a plain str at runtime. The hasattr/.value branch is defensive-only and will
    # not fire under normal operation, but guards against future config schema changes.
    fw = config.framework
    return fw.value if hasattr(fw, "value") else str(fw)  # type: ignore[union-attr]
```

7. **M9** — Change `_render_stages` signature to accept `fw`:
```python
def _render_stages(
    config: RAGPipelineConfig,
    loader: TemplateLoader,
    fw: str,           # ← added
) -> dict[str, str]:
```
Remove the local `fw = _framework_str(config)` inside `_render_stages` (line 262). In `generate()`, the existing `fw = _framework_str(config)` (line 358) is already computed; pass it: `stages = _render_stages(config, loader, fw)`.

8. **M5** — Above `base_ctx` in `_render_stages`, add comment:
```python
# Template context contract:
#   - Per-stage templates (stages/<cat>/<type>.py.j2) should read the
#     unpacked sub-object ("chunking", "embedding", "vector_db",
#     "retrieval", "reranker", "llm") — NOT reach into `config`.
#   - Entrypoint templates (pipeline.py.j2, ingestion.py.j2) read `config`
#     for cross-cutting concerns (name, framework, dependencies).
#   - `config` is passed into stage context for rare cross-cutting reads only;
#     use the sub-object key as the primary access path in stage templates.
```

**Verification:** `python -m pytest tests/ -q`  
**Expected test count:** unchanged from Phase 2 (≥ 270).

---

### Phase 4 — GeneratorResult property + test hygiene (M8, m4, C2)

**Goal:** Eliminate dual representation; fix flaky-cwd test pattern; add stub-template meta-coverage.

**Files touched (2):**
- `ragfactory/core/generator.py`
- `tests/unit/test_generator.py`

#### Changes to `generator.py`

**M8** — Replace `GeneratorResult` dataclass:
```python
@dataclass
class GeneratorResult:
    """
    Result of a generate() call.

    .generated_files    — canonical list of GeneratedFile (source of truth)
    .files              — convenience dict[path → content], computed from generated_files
    .validation_passed  — False if any ast.parse() error occurred
    .errors             — list of ast.parse error strings (empty on success)
    .config_yaml        — the input config round-tripped to YAML
    """
    generated_files:   list[GeneratedFile] = field(default_factory=list)
    validation_passed: bool = True
    errors:            list[str] = field(default_factory=list)
    config_yaml:       str = ""

    @property
    def files(self) -> dict[str, str]:
        """Convenience dict built from generated_files. Read-only."""
        return {gf.path: gf.content for gf in self.generated_files}
```

Update `generate()` internals:
- Delete local `files: dict[str, str] = {}` (line 355).
- Remove every `files[path] = content` line in both Python and non-Python file loops.
- In the `except GeneratorError` block, remove `files={}` from the constructor call.
- In the final `return GeneratorResult(...)`, remove `files=files`.
- `generated_files` is unchanged — still the canonical store.

#### Changes to `tests/unit/test_generator.py`

**m4** — In `TestPureFunction.test_no_files_written_to_cwd`, add `monkeypatch` parameter and replace `os.chdir` block:
```python
def test_no_files_written_to_cwd(
    self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _gen()
    written = list(tmp_path.iterdir())
    assert written == [], f"Generator wrote files to disk: {written}"
```
Remove `import os` if it's the only use of `os` in the file — verify first.

**C2** — Add `class TestStubTemplateCoverage` at the bottom of the file:
```python
import typing
from ragfactory.core import config as cfg_mod

def _union_type_literals(union_annotation) -> list[str]:
    """Extract the `type` field default from every member of a discriminated union."""
    result = []
    for member in typing.get_args(union_annotation):
        if hasattr(member, "model_fields") and "type" in member.model_fields:
            result.append(member.model_fields["type"].default)
    return result

_STUB_STAGE_UNIONS = [
    (cfg_mod.ChunkingConfig,  "stages/chunking"),
    (cfg_mod.EmbeddingConfig, "stages/embedding"),
    (cfg_mod.VectorDBConfig,  "stages/vectordb"),
    (cfg_mod.RetrievalConfig, "stages/retrieval"),
    (cfg_mod.RerankerConfig,  "stages/reranker"),
    (cfg_mod.LLMConfig,       "stages/llm"),
]

_stub_params = [
    (subdir, type_literal)
    for union, subdir in _STUB_STAGE_UNIONS
    for type_literal in _union_type_literals(union)
]

class TestStubTemplateCoverage:
    """Meta-test: every config type literal must have a stub .j2 file."""

    @pytest.mark.parametrize("subdir,type_literal", _stub_params,
                              ids=[f"{s}/{t}" for s, t in _stub_params])
    def test_stub_exists(self, subdir: str, type_literal: str) -> None:
        stub = STUB_TEMPLATE_DIR / subdir / f"{type_literal}.py.j2"
        assert stub.exists(), (
            f"Missing stub template: {stub}\n"
            f"Add a minimal stub at tests/fixtures/stub_templates/{subdir}/{type_literal}.py.j2"
        )
```

Note: If any stub is missing, the test fails with a clear message naming the missing file. **Do not add missing stubs in this phase** — report to the user first.

**Verification:** `python -m pytest tests/ -q`  
**Expected test count:** ≥ 270 + one per type literal in the 6 unions (count by summing `len(_union_type_literals(...))` for each union: 7 chunking + 7 embedding + 6 vectordb + 5 retrieval + 4 reranker + 4 llm = 33 new meta-tests). Target: ≥ 303. Must not decrease.

---

### Phase 5 — Package template files (m12 completion)

**Goal:** Ensure `importlib.resources.files("ragfactory") / "templates"` works when installed from a wheel.

**Files touched (1):**
- `pyproject.toml`

**Changes:**

In the `[tool.hatch.build.targets.wheel]` section (or add it if absent), include:
```toml
[tool.hatch.build.targets.wheel]
packages = ["ragfactory"]
include = ["ragfactory/templates/**"]
# Templates must be bundled — ragfactory/core/generator.py uses
# importlib.resources.files("ragfactory") / "templates" which requires
# template files to be present inside the installed package.
```

If `force-include` syntax is preferred:
```toml
[tool.hatch.build.targets.wheel.force-include]
"ragfactory/templates" = "ragfactory/templates"
```
Pick whichever form the current `pyproject.toml` already uses for package data.

**Verification:**
- `python -m pytest tests/ -q` — must pass (unchanged).
- Visual check: `python -m build --wheel && unzip -l dist/*.whl | grep templates` to confirm `ragfactory/templates/**/*.j2` entries appear in the wheel.

**Expected test count:** unchanged from Phase 4.

---

## 4. Risk Register

| Phase | Risk | Detection |
|---|---|---|
| 0 | Deleting calls in `validate()` accidentally removes wrong lines | Full test run; grep `_check_hybrid_search_vectordb\|_check_sentence_window_framework` must return 0 hits after. |
| 1 | `_providers.py` accidentally imports `ragfactory.core.*` creating a circular import | `python -c "import ragfactory.core.validator, ragfactory.core.generator"` must succeed without `ImportError`. |
| 2 | M3 regression: bare Ollama models (`llama3.2`) stop emitting slow-local warning | Dedicated `TestContextualChunkingCloudModels` test for `llama3.2`. |
| 2 | M1 tests over-constrain `_is_active` — fail on future config schema changes | Tests assert on discriminator literals from the actual config model, not hard-coded strings. Use parametrize where practical. |
| 3 | M4 changes error messages for `render_entrypoint`/`render_common` — tests assert `match="Template not found"` which still passes, but any downstream parser of the message text would see extra text | No downstream message parsers in repo. Change makes messages MORE informative, not less. |
| 3 | m12: `importlib.resources` returns a `MultiplexedPath` (namespace package) that `str(...)` can't coerce | `ragfactory` is a regular package with `__init__.py`, not a namespace package. Verify by running the test suite after Phase 3. |
| 3 | M9: caller of `_render_stages` outside `generate()` breaks | No external callers — `_render_stages` is private. Tests call `render_stage` (public), not `_render_stages`. |
| 4 | M8 breaks callers constructing `GeneratorResult(files=...)` | Grep the repo for `GeneratorResult(` — only `generate()` constructs it. Tests only read `.files`. |
| 4 | C2 discovers missing stub templates → test fails | Expected. Stop and report to user; do not add stubs within this phase. |
| 5 | Hatch `force-include` vs `include` syntax varies by version | Verify with a clean wheel build; fall back to the other form if one fails. |

---

## 5. Contract Stability Check

**`validate(config, corpus_tokens) → ValidationResult`** — signature unchanged across all phases.  
`ValidationResult` fields (`.valid`, `.issues`, `.costs`, `.errors`, `.warnings`, `.infos`) unchanged.  
New issue codes emitted for Gemini/Mistral/Cohere context models are net-additions, not renames.

**`generate(config, template_dir) → GeneratorResult`** — signature unchanged.  
`GeneratorResult.files` remains accessible as a `dict[str, str]` (now via `@property`).  
`.generated_files`, `.validation_passed`, `.errors`, `.config_yaml` unchanged.  
**Only breaking change:** constructor kwarg `files=` removed. No code in the repo constructs `GeneratorResult` directly with `files=` — verified.

**`ragfactory/core/_providers.py`** — leading underscore = explicitly internal, not part of the public API.

**Test count invariant:** 227 → ≥ 270 (after Phase 2) → ≥ 303 (after Phase 4). Only grows, never decreases. Verified at the end of every phase via `python -m pytest tests/ -q`.
