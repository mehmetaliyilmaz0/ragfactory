"""
Compatibility data for RAGFactory pipeline configurations.

Design principles:
  - Pure data module. Zero logic. Zero project imports.
  - Single source of truth for all compatibility knowledge.
  - CLI, validator, Phase 2 API, and Phase 3 UI all read from here.
  - Importable in isolation — no circular dependency risk.
  - Frozen dataclasses: these are compile-time constants, not runtime models.

Three data structures:
  INCOMPATIBLE      — hard failures (pipeline will not function)
  WARNINGS          — soft warnings (cost, performance, operational concerns)
  CROSS_FIELD_RULES — rules requiring field value inspection (metadata only;
                      logic lives in validator.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ─── Types ────────────────────────────────────────────────────────────────────


class Severity(str, Enum):
    INFO       = "info"        # informational, no action required
    WARNING    = "warning"     # may cause issues or suboptimal performance
    COST_ALERT = "cost_alert"  # significant cost implication


@dataclass(frozen=True)
class IncompatiblePair:
    """Two components that cannot be used together in a pipeline."""

    component_a: str       # dot-path, e.g. "generation.advanced.flare"
    component_b: str       # dot-path or "*" for unconditional block
    reason: str            # human-readable error message
    doc_url: str | None    # optional upstream issue / docs link


@dataclass(frozen=True)
class CompatibilityWarning:
    """A condition that warrants a warning but does not block generation."""

    condition: str                  # dot-path or compound condition identifier
    message: str                    # may contain {placeholders} for interpolation
    severity: Severity
    cost_per_million: float | None  # USD per 1M tokens; None if not cost-related


@dataclass(frozen=True)
class CrossFieldRule:
    """
    A rule that requires inspecting field *values*, not just component types.
    Metadata only — the actual check logic lives in validator.py.
    """

    rule_id: str
    description: str


# ─── Hard Incompatibilities ───────────────────────────────────────────────────
#
# Dot-path format: "section.subsection.type_value"
#   Last segment = discriminator `type` value  OR  ".enabled" for boolean flags.
#
# Examples:
#   "generation.llm.anthropic"       → config.generation.llm.type == "anthropic"
#   "generation.advanced.flare"      → advanced.flare is not None and .enabled
#   "indexing.chunking.late"         → config.indexing.chunking.type == "late"
#   "retrieval.hybrid_rrf"           → config.retrieval.type == "hybrid_rrf"
#   "framework.langchain"            → config.framework == "langchain"

INCOMPATIBLE: list[IncompatiblePair] = [
    # ── FLARE × LLM provider ─────────────────────────────────────────────────
    # FLARE (Forward-Looking Active Retrieval) monitors per-token generation
    # confidence and triggers retrieval when probability drops below threshold.
    # This requires the LLM to expose token-level logprobs — most providers don't.
    IncompatiblePair(
        component_a="generation.advanced.flare",
        component_b="generation.llm.anthropic",
        reason=(
            "FLARE requires token-level logprobs to monitor generation confidence. "
            "The Anthropic API does not expose logprobs. "
            "Use OpenAI (gpt-4o / gpt-4o-mini) which supports logprobs=True."
        ),
        doc_url="https://arxiv.org/abs/2305.06983",
    ),
    IncompatiblePair(
        component_a="generation.advanced.flare",
        component_b="generation.llm.cohere_llm",
        reason=(
            "FLARE requires token-level logprobs. "
            "Cohere Command-R / Command-R+ does not expose per-token log probabilities "
            "through its API. Use OpenAI instead."
        ),
        doc_url=None,
    ),

    # ── Late Chunking × Embedding provider ───────────────────────────────────
    # Late chunking (Jina AI, 2024) embeds the full document first, then pools
    # token embeddings per chunk span. This is architecturally specific to Jina's
    # embedding models and cannot be replicated with other providers.
    IncompatiblePair(
        component_a="indexing.chunking.late",
        component_b="indexing.embedding.openai",
        reason=(
            "Late chunking requires Jina embedding models with late_chunking=True. "
            "OpenAI text-embedding models do not support late chunking. "
            "Switch to embedding.type='jina' with model='jina-embeddings-v3'."
        ),
        doc_url="https://jina.ai/news/late-chunking-in-long-context-embedding-models/",
    ),
    IncompatiblePair(
        component_a="indexing.chunking.late",
        component_b="indexing.embedding.cohere",
        reason=(
            "Late chunking is a Jina-specific architecture. "
            "Cohere embed models do not support late chunking. "
            "Switch to embedding.type='jina' with model='jina-embeddings-v3'."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="indexing.chunking.late",
        component_b="indexing.embedding.voyage",
        reason=(
            "Late chunking is a Jina-specific architecture. "
            "Voyage AI embedding models do not support late chunking. "
            "Switch to embedding.type='jina' with model='jina-embeddings-v3'."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="indexing.chunking.late",
        component_b="indexing.embedding.gemini",
        reason=(
            "Late chunking is a Jina-specific architecture. "
            "Google Gemini embedding models do not support late chunking. "
            "Switch to embedding.type='jina' with model='jina-embeddings-v3'."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="indexing.chunking.late",
        component_b="indexing.embedding.bge_m3",
        reason=(
            "Late chunking is a Jina-specific architecture. "
            "BGE-M3 does not support late chunking despite being a multi-vector model. "
            "Switch to embedding.type='jina' with model='jina-embeddings-v3'."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="indexing.chunking.late",
        component_b="indexing.embedding.nomic",
        reason=(
            "Late chunking is a Jina-specific architecture. "
            "Nomic embed-text does not support late chunking. "
            "Switch to embedding.type='jina' with model='jina-embeddings-v3'."
        ),
        doc_url=None,
    ),

    # ── Hybrid Search × Vector DB ─────────────────────────────────────────────
    # Hybrid search (BM25 + dense) requires the vector DB to support sparse vectors
    # natively, or a separate BM25 index alongside the dense index.
    # Phase 1c templates only implement hybrid for DBs with native support.
    IncompatiblePair(
        component_a="retrieval.hybrid_rrf",
        component_b="indexing.vector_db.chromadb",
        reason=(
            "ChromaDB does not have native sparse/BM25 vector support. "
            "Hybrid RRF retrieval requires a vector DB with built-in sparse indexing. "
            "Switch to qdrant, weaviate, milvus, or pgvector."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="retrieval.hybrid_weighted",
        component_b="indexing.vector_db.chromadb",
        reason=(
            "ChromaDB does not have native sparse/BM25 vector support. "
            "Hybrid weighted retrieval requires a vector DB with built-in sparse indexing. "
            "Switch to qdrant, weaviate, milvus, or pgvector."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="retrieval.hybrid_rrf",
        component_b="indexing.vector_db.pinecone",
        reason=(
            "The langchain-pinecone and llama-index-vector-stores-pinecone integrations "
            "do not reliably expose Pinecone's sparse-dense API. "
            "Hybrid RRF is blocked until the integration matures. "
            "Switch to qdrant, weaviate, milvus, or pgvector for hybrid search."
        ),
        doc_url=None,
    ),
    IncompatiblePair(
        component_a="retrieval.hybrid_weighted",
        component_b="indexing.vector_db.pinecone",
        reason=(
            "The langchain-pinecone and llama-index-vector-stores-pinecone integrations "
            "do not reliably expose Pinecone's sparse-dense API. "
            "Hybrid weighted retrieval is blocked until the integration matures. "
            "Switch to qdrant, weaviate, milvus, or pgvector for hybrid search."
        ),
        doc_url=None,
    ),

    # ── Sentence Window × Framework ───────────────────────────────────────────
    # SentenceWindowNodeParser + MetadataReplacementPostProcessor are LlamaIndex-
    # native patterns. LangChain has no direct equivalent.
    IncompatiblePair(
        component_a="retrieval.sentence_window",
        component_b="framework.langchain",
        reason=(
            "Sentence Window Retrieval uses LlamaIndex-native components "
            "(SentenceWindowNodeParser, MetadataReplacementPostProcessor). "
            "No equivalent exists in LangChain. "
            "Switch to framework='llamaindex' or use a different retrieval strategy."
        ),
        doc_url=None,
    ),
]


# ─── Soft Warnings ────────────────────────────────────────────────────────────

WARNINGS: list[CompatibilityWarning] = [
    # Cost alerts
    CompatibilityWarning(
        condition="indexing.chunking.contextual",
        message=(
            "Contextual chunking makes one LLM call per chunk to generate a context "
            "description. Estimated cost: ~$1.02/M document tokens with prompt caching "
            "($5.10/M without caching). Budget accordingly for large corpora."
        ),
        severity=Severity.COST_ALERT,
        cost_per_million=1.02,
    ),
    CompatibilityWarning(
        condition="indexing.chunking.proposition",
        message=(
            "Proposition chunking makes one LLM call per paragraph to extract atomic "
            "propositions. Estimated cost: ~$2.50/M document tokens with gpt-4o-mini "
            "($12.50/M with gpt-4o). Budget accordingly."
        ),
        severity=Severity.COST_ALERT,
        cost_per_million=2.50,
    ),
    CompatibilityWarning(
        condition="generation.advanced.agentic",
        message=(
            "Agentic RAG uses up to max_reasoning_steps LLM calls per query. "
            "Expect 3–10x query cost vs. simple RAG. Use only for complex, "
            "multi-step tasks where accuracy justifies the cost."
        ),
        severity=Severity.COST_ALERT,
        cost_per_million=None,
    ),
    CompatibilityWarning(
        condition="evaluation",
        message=(
            "Pipeline evaluation with a judge LLM adds significant cost. "
            "A 50-sample RAGAS run with GPT-4o costs approximately $2–5 per run. "
            "Consider using gpt-4o-mini as judge for development runs."
        ),
        severity=Severity.COST_ALERT,
        cost_per_million=None,
    ),

    # Operational warnings
    CompatibilityWarning(
        condition="generation.advanced.crag.web_search_fallback",
        message=(
            "CRAG web search fallback adds 1–3 seconds latency per low-confidence query. "
            "Disable web_search_fallback=False in air-gapped or latency-sensitive "
            "environments."
        ),
        severity=Severity.WARNING,
        cost_per_million=None,
    ),
    CompatibilityWarning(
        condition="indexing.embedding.bge_m3",
        message=(
            "BGE-M3 is self-hosted. CPU inference is 10–50x slower than GPU. "
            "Ensure your deployment environment has GPU resources for production "
            "throughput. Add a GPU node to your docker-compose or Kubernetes spec."
        ),
        severity=Severity.WARNING,
        cost_per_million=None,
    ),
    CompatibilityWarning(
        condition="post_retrieval.reranker.cross_encoder",
        message=(
            "Cross-encoder reranker requires GPU for production throughput (>10 QPS). "
            "CPU inference is viable only at low query rates (<5–10 QPS). "
            "Consider Cohere Rerank API or FlashRank for CPU-only deployments."
        ),
        severity=Severity.WARNING,
        cost_per_million=None,
    ),
    CompatibilityWarning(
        condition="indexing.vector_db.chromadb",
        message=(
            "ChromaDB is an embedded, in-process vector store optimised for prototyping. "
            "Maximum tested scale: ~7M vectors. Not recommended for production workloads. "
            "Switch to qdrant (production default) before going live."
        ),
        severity=Severity.WARNING,
        cost_per_million=None,
    ),

    # Informational
    CompatibilityWarning(
        condition="post_retrieval.reranker.colbert",
        message=(
            "ColBERT (via RAGatouille) requires significant disk space: "
            "6–10x the document size after ColBERTv2 compression. "
            "Plan storage accordingly before indexing large corpora."
        ),
        severity=Severity.INFO,
        cost_per_million=None,
    ),
    CompatibilityWarning(
        condition="pre_retrieval.hyde",
        message=(
            "HyDE adds one LLM call per query for hypothetical document generation. "
            "Small latency and cost increase per query (~1 LLM call overhead)."
        ),
        severity=Severity.INFO,
        cost_per_million=None,
    ),
]


# ─── Cross-Field Rules ────────────────────────────────────────────────────────
# Rules that require inspecting field *values* (not just types) across models.
# Metadata only — check logic is in validator.py.

CROSS_FIELD_RULES: list[CrossFieldRule] = [
    CrossFieldRule(
        rule_id="late_chunking_jina_v3_flag",
        description=(
            "When chunking.type=='late': embedding must be jina, "
            "model=='jina-embeddings-v3', and late_chunking==True. "
            "All three conditions are required."
        ),
    ),
    CrossFieldRule(
        rule_id="late_chunking_jina_v2",
        description=(
            "jina-embeddings-v2-base-en does not support late chunking "
            "(requires 8K context window available only in v3). ERROR."
        ),
    ),
    CrossFieldRule(
        rule_id="contextual_chunking_throughput",
        description=(
            "When chunking.type=='contextual' and context_model is an Ollama-style "
            "local model: technically feasible but ~100x slower than Claude Haiku "
            "for large corpora. WARNING."
        ),
    ),
    CrossFieldRule(
        rule_id="contextual_extra_api_key",
        description=(
            "When chunking.type=='contextual' and context_model provider differs "
            "from the main LLM provider: an extra API key is required. INFO."
        ),
    ),
    CrossFieldRule(
        rule_id="multiple_advanced_techniques",
        description=(
            "At most one of crag, flare, agentic should be enabled simultaneously. "
            "Pydantic allows multiple but combined behaviour is undefined. WARNING."
        ),
    ),
    CrossFieldRule(
        rule_id="reranker_top_n_vs_top_k",
        description=(
            "reranker.top_n >= retrieval.top_k means the reranker receives fewer "
            "documents than it expects to select. Almost always a user mistake. WARNING."
        ),
    ),
]
