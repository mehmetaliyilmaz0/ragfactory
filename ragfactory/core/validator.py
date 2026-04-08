"""
Compatibility validator for RAGFactory pipeline configurations.

Design principles:
  - Single public function: validate(config, corpus_tokens) -> ValidationResult
  - Pure function — no side effects, no I/O, no mutations
  - All logic lives here; compatibility DATA lives in compatibility.py
  - Dataclass outputs (not Pydantic) — ValidationResult is output, not user input
  - Each checker is a private function that appends to shared issue/cost lists

Usage:
    from ragfactory.core.config import RAGPipelineConfig
    from ragfactory.core.validator import validate

    result = validate(config)
    if not result.valid:
        for err in result.errors:
            print(f"ERROR [{err.code}]: {err.message}")

    result_with_cost = validate(config, corpus_tokens=5_000_000)
    for cost in result_with_cost.costs:
        print(f"{cost.component}: ${cost.estimated_total:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ragfactory.core._providers import (
    PROVIDER_ENV_VAR,
    infer_context_model_provider,
    is_probably_local_model,
)
from ragfactory.core.compatibility import INCOMPATIBLE, WARNINGS, Severity
from ragfactory.core.config import (
    RAGPipelineConfig,
)


# ─── Output Types ─────────────────────────────────────────────────────────────


class ValidationSeverity(str, Enum):
    ERROR   = "error"    # pipeline will not function at runtime
    WARNING = "warning"  # may cause issues or suboptimal performance
    INFO    = "info"     # informational, no action required
    COST    = "cost"     # cost estimation output


@dataclass
class ValidationIssue:
    severity:       ValidationSeverity
    code:           str             # machine-readable, e.g. "FLARE_NO_LOGPROBS"
    message:        str             # human-readable description
    component_path: str             # dot-path to offending config field
    suggestion:     str | None = None  # recommended remediation


@dataclass
class CostEstimate:
    component:               str
    description:             str
    cost_per_million_tokens: float
    estimated_total:         float | None  # None when corpus_tokens not provided


@dataclass
class ValidationResult:
    """
    Result of a validate() call.

    .valid is True iff there are zero ERROR-severity issues.
    Warnings and cost alerts do not affect .valid.
    """

    valid:  bool
    issues: list[ValidationIssue] = field(default_factory=list)
    costs:  list[CostEstimate]    = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]


# ─── Cost table ───────────────────────────────────────────────────────────────
# Maps a check key → (component label, USD per 1M tokens).
# Populated from WARNINGS cost_per_million where applicable.

_COST_TABLE: dict[str, tuple[str, float]] = {
    "contextual_chunking": (
        "Contextual chunk annotation (with prompt caching)",
        1.02,
    ),
    "proposition_chunking": (
        "Proposition extraction via gpt-4o-mini",
        2.50,
    ),
}


# ─── Dot-path resolution ──────────────────────────────────────────────────────

def _is_active(config: RAGPipelineConfig, path: str) -> bool:
    """
    Resolve a dot-path like "generation.advanced.flare" or "indexing.chunking.late"
    against a config object and return True if the described component is active.

    Path format: "section[.subsection]*.type_value_or_flag"

    The last segment is matched against the discriminator `type` field value,
    or for advanced-generation features, checks `.enabled == True`.
    """
    # fmt: off
    p = path

    # ── framework ─────────────────────────────────────────────────────────
    if p == "framework.langchain":
        return config.framework == "langchain"
    if p == "framework.llamaindex":
        return config.framework == "llamaindex"

    # ── indexing.chunking ─────────────────────────────────────────────────
    if p.startswith("indexing.chunking."):
        return config.indexing.chunking.type == p.removeprefix("indexing.chunking.")

    # ── indexing.embedding ────────────────────────────────────────────────
    if p.startswith("indexing.embedding."):
        return config.indexing.embedding.type == p.removeprefix("indexing.embedding.")

    # ── indexing.vector_db ────────────────────────────────────────────────
    if p.startswith("indexing.vector_db."):
        return config.indexing.vector_db.type == p.removeprefix("indexing.vector_db.")

    # ── retrieval ─────────────────────────────────────────────────────────
    if p.startswith("retrieval."):
        return config.retrieval.type == p.removeprefix("retrieval.")  # type: ignore[union-attr]

    # ── pre_retrieval ─────────────────────────────────────────────────────
    if p == "pre_retrieval.hyde":
        return config.pre_retrieval.hyde is not None and config.pre_retrieval.hyde.enabled
    if p == "pre_retrieval.query_rewriting":
        return (config.pre_retrieval.query_rewriting is not None
                and config.pre_retrieval.query_rewriting.enabled)
    if p == "pre_retrieval.routing":
        return config.pre_retrieval.routing is not None and config.pre_retrieval.routing.enabled

    # ── post_retrieval.reranker ───────────────────────────────────────────
    if p.startswith("post_retrieval.reranker."):
        if config.post_retrieval.reranker is None:
            return False
        return config.post_retrieval.reranker.type == p.removeprefix("post_retrieval.reranker.")

    # ── generation.llm ────────────────────────────────────────────────────
    if p.startswith("generation.llm."):
        return config.generation.llm.type == p.removeprefix("generation.llm.")

    # ── generation.advanced.* ─────────────────────────────────────────────
    if p == "generation.advanced.flare":
        adv = config.generation.advanced
        return adv is not None and adv.flare is not None and adv.flare.enabled

    if p == "generation.advanced.crag":
        adv = config.generation.advanced
        return adv is not None and adv.crag is not None and adv.crag.enabled

    if p == "generation.advanced.agentic":
        adv = config.generation.advanced
        return adv is not None and adv.agentic is not None and adv.agentic.enabled

    if p == "generation.advanced.crag.web_search_fallback":
        adv = config.generation.advanced
        return (adv is not None and adv.crag is not None
                and adv.crag.enabled and adv.crag.web_search_fallback)

    # ── evaluation ────────────────────────────────────────────────────────
    if p == "evaluation":
        return config.evaluation is not None

    # ── ingestion.parser ──────────────────────────────────────────────────
    if p.startswith("ingestion.parser."):
        return config.ingestion.parser == p.removeprefix("ingestion.parser.")

    # fmt: on
    return False


# ─── Checker functions ────────────────────────────────────────────────────────

def _check_incompatible_pairs(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """Iterate INCOMPATIBLE and emit an ERROR for every active pair."""
    for pair in INCOMPATIBLE:
        if _is_active(config, pair.component_a) and _is_active(config, pair.component_b):
            # Build a short machine-readable code from the pair paths
            a_tail = pair.component_a.rsplit(".", 1)[-1].upper()
            b_tail = pair.component_b.rsplit(".", 1)[-1].upper()
            code = f"INCOMPAT_{a_tail}_{b_tail}"

            suggestion = pair.doc_url and f"See: {pair.doc_url}"

            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=code,
                message=pair.reason,
                component_path=pair.component_a,
                suggestion=suggestion,
            ))


def _check_late_chunking(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """
    Cross-field checks for late chunking that go beyond the INCOMPATIBLE pairs.
    The INCOMPATIBLE list catches wrong embedding provider; this checker catches
    wrong Jina model version and missing late_chunking flag.
    """
    if config.indexing.chunking.type != "late":
        return

    emb = config.indexing.embedding
    if emb.type != "jina":
        # Already caught by INCOMPATIBLE — skip to avoid duplicate error
        return

    # At this point embedding IS jina — check model version and flag
    if emb.model == "jina-embeddings-v2-base-en":  # type: ignore[union-attr]
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="LATE_CHUNKING_JINA_V2",
            message=(
                "Late chunking requires jina-embeddings-v3 (8K context window). "
                "jina-embeddings-v2-base-en does not support late chunking."
            ),
            component_path="indexing.embedding.model",
            suggestion="Set embedding.model='jina-embeddings-v3'.",
        ))

    if not emb.late_chunking:  # type: ignore[union-attr]
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="LATE_CHUNKING_FLAG_NOT_SET",
            message=(
                "chunking.type='late' is selected but embedding.late_chunking=False. "
                "Set embedding.late_chunking=True to enable late chunking in the Jina API."
            ),
            component_path="indexing.embedding.late_chunking",
            suggestion="Set indexing.embedding.late_chunking=True.",
        ))


def _check_contextual_chunking(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """
    Cross-field checks for contextual chunking.
    Emits a throughput WARNING when context_model is a local Ollama-style model,
    and an INFO notice when the context model requires an extra API key beyond
    the pipeline LLM. Provider detection is delegated to _providers.py.
    """
    if config.indexing.chunking.type != "contextual":
        return

    chunking = config.indexing.chunking
    context_model: str = chunking.context_model  # type: ignore[union-attr]

    context_provider = infer_context_model_provider(context_model)

    if context_provider is None:
        # Unrecognised provider. Check if it looks like a bare local Ollama model.
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
        # Unknown cloud-like model (contains '/' or ':') — emit nothing.
        return

    # Known cloud provider: check if it needs an extra API key beyond the main LLM.
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


def _check_flare_logprobs(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """
    FLARE × LLM logprobs check.
    INCOMPATIBLE already covers Anthropic and Cohere (hard errors).
    This checker adds a WARNING for Ollama (logprob support is model-dependent).
    """
    if not _is_active(config, "generation.advanced.flare"):
        return

    if config.generation.llm.type == "ollama":
        model_name = config.generation.llm.model  # type: ignore[union-attr]
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="FLARE_OLLAMA_LOGPROBS_UNRELIABLE",
            message=(
                f"FLARE requires token-level logprobs. "
                f"Ollama model '{model_name}' may expose logprobs depending on the model, "
                "but langchain-ollama and llama-index-llms-ollama do not reliably surface them. "
                "FLARE will likely fail at runtime. "
                "Test thoroughly before deploying."
            ),
            component_path="generation.llm.type",
            suggestion="Switch to OpenAI (gpt-4o / gpt-4o-mini) which reliably supports logprobs.",
        ))


def _check_multiple_advanced_techniques(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """
    Warn when more than one advanced generation technique is enabled simultaneously.
    Combined behaviour (e.g. CRAG + FLARE) is undefined and likely buggy.
    """
    adv = config.generation.advanced
    if adv is None:
        return

    enabled_techniques: list[str] = []
    if adv.crag is not None and adv.crag.enabled:
        enabled_techniques.append("crag")
    if adv.flare is not None and adv.flare.enabled:
        enabled_techniques.append("flare")
    if adv.agentic is not None and adv.agentic.enabled:
        enabled_techniques.append("agentic")

    if len(enabled_techniques) > 1:
        names = ", ".join(enabled_techniques)
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="MULTIPLE_ADVANCED_TECHNIQUES",
            message=(
                f"Multiple advanced generation techniques enabled simultaneously: {names}. "
                "Combined behaviour is undefined — at most one should be active at a time. "
                "Disable all but one."
            ),
            component_path="generation.advanced",
            suggestion=f"Keep only one of: {names}.",
        ))


def _check_reranker_top_n(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """
    Warn when reranker.top_n >= retrieval.top_k.
    The reranker cannot select more documents than were retrieved.
    """
    reranker = config.post_retrieval.reranker
    if reranker is None:
        return

    retrieval_top_k: int | None = getattr(config.retrieval, "top_k", None)
    if retrieval_top_k is None:
        return

    if reranker.top_n >= retrieval_top_k:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="RERANKER_TOP_N_EXCEEDS_TOP_K",
            message=(
                f"reranker.top_n={reranker.top_n} >= retrieval.top_k={retrieval_top_k}. "
                "The reranker will receive fewer documents than it expects to select. "
                "This is almost always a misconfiguration."
            ),
            component_path="post_retrieval.reranker.top_n",
            suggestion=(
                f"Set retrieval.top_k > reranker.top_n. "
                f"Typical: top_k=20–100, top_n=3–10."
            ),
        ))


def _check_warnings(
    config: RAGPipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    """
    Iterate WARNINGS from compatibility.py and emit INFO/WARNING/COST_ALERT issues
    for every active condition.
    """
    _severity_map = {
        Severity.INFO:       ValidationSeverity.INFO,
        Severity.WARNING:    ValidationSeverity.WARNING,
        Severity.COST_ALERT: ValidationSeverity.COST,
    }

    for w in WARNINGS:
        if not _is_active(config, w.condition):
            continue

        tail = w.condition.rsplit(".", 1)[-1].upper()
        code = f"WARN_{tail}"

        issues.append(ValidationIssue(
            severity=_severity_map[w.severity],
            code=code,
            message=w.message,
            component_path=w.condition,
            suggestion=None,
        ))


def _estimate_costs(
    config: RAGPipelineConfig,
    corpus_tokens: int | None,
    costs: list[CostEstimate],
) -> None:
    """
    Produce cost estimates for components with known per-token rates.
    """
    active_checks: dict[str, bool] = {
        "contextual_chunking": config.indexing.chunking.type == "contextual",
        "proposition_chunking": config.indexing.chunking.type == "proposition",
    }

    for key, is_active in active_checks.items():
        if not is_active:
            continue
        description, rate = _COST_TABLE[key]
        estimated_total: float | None = None
        if corpus_tokens is not None:
            estimated_total = rate * corpus_tokens / 1_000_000

        costs.append(CostEstimate(
            component=key,
            description=description,
            cost_per_million_tokens=rate,
            estimated_total=estimated_total,
        ))


# ─── Public API ───────────────────────────────────────────────────────────────

def validate(
    config: RAGPipelineConfig,
    corpus_tokens: int | None = None,
) -> ValidationResult:
    """
    Run all compatibility checks and cost estimates on a RAGPipelineConfig.

    Args:
        config:        A validated RAGPipelineConfig instance.
        corpus_tokens: Optional estimated corpus size in tokens. When provided,
                       cost estimates include an absolute dollar amount.
                       When None, only per-million rates are returned.

    Returns:
        ValidationResult with .valid == True iff there are zero ERROR issues.

    Raises:
        TypeError: If config is not a RAGPipelineConfig instance.

    Example:
        result = validate(my_config)
        if not result.valid:
            for err in result.errors:
                print(f"[{err.code}] {err.message}")
    """
    if not isinstance(config, RAGPipelineConfig):
        raise TypeError(
            f"validate() expects a RAGPipelineConfig, got {type(config).__name__}"
        )

    issues: list[ValidationIssue] = []
    costs:  list[CostEstimate]    = []

    # Run all checkers in order
    _check_incompatible_pairs(config, issues)
    _check_late_chunking(config, issues)
    _check_contextual_chunking(config, issues)
    _check_flare_logprobs(config, issues)
    _check_multiple_advanced_techniques(config, issues)
    _check_reranker_top_n(config, issues)
    _check_warnings(config, issues)
    _estimate_costs(config, corpus_tokens, costs)

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues, costs=costs)
