"""
Core configuration models for RAGFactory pipeline definitions.

Design principles:
  - Discriminated unions: each sub-config carries only its own valid fields.
    No "alpha" field on a dense retriever. No "rrf_k" on a weighted retriever.
  - StrictModel: extra fields are forbidden — typos in YAML surface immediately.
  - All secrets (API keys, connection strings) come from env vars, never from config.
  - JSON Schema export works out of the box via model_json_schema() — used by Phase 3 UI.
  - Full YAML round-trip: config → YAML file → config, bit-for-bit equivalent.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


class StrictModel(BaseModel):
    """Base model that forbids extra fields — typos in YAML surface as errors."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Framework(str, Enum):
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────


class FileSourceConfig(StrictModel):
    type: Literal["file"] = "file"
    path: str = Field(..., description="Path to a file or directory")
    glob: str = Field("**/*", description="Glob pattern when path is a directory")
    recursive: bool = True


class S3SourceConfig(StrictModel):
    type: Literal["s3"] = "s3"
    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    # Credentials from env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


class URLSourceConfig(StrictModel):
    type: Literal["url"] = "url"
    urls: list[str] = Field(..., min_length=1)


SourceConfig = Annotated[
    Union[FileSourceConfig, S3SourceConfig, URLSourceConfig],
    Field(discriminator="type"),
]


class ParserType(str, Enum):
    DEFAULT = "default"
    UNSTRUCTURED = "unstructured"
    AZURE_DOC_INTELLIGENCE = "azure_doc_intelligence"
    DOCLING = "docling"


class IngestionConfig(StrictModel):
    sources: list[SourceConfig] = Field(default_factory=list)
    parser: ParserType = ParserType.DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────


class FixedChunkingConfig(StrictModel):
    type: Literal["fixed"] = "fixed"
    chunk_size: int = Field(512, ge=64, le=8192, description="Tokens per chunk")
    chunk_overlap: int = Field(50, ge=0, description="Overlap tokens between adjacent chunks")

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> "FixedChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class RecursiveChunkingConfig(StrictModel):
    type: Literal["recursive"] = "recursive"
    chunk_size: int = Field(512, ge=64, le=8192)
    chunk_overlap: int = Field(50, ge=0)
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""],
        description="Separator hierarchy — tried in order until chunk fits",
    )

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> "RecursiveChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class BreakpointType(str, Enum):
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    INTERQUARTILE = "interquartile"
    GRADIENT = "gradient"


class SemanticChunkingConfig(StrictModel):
    type: Literal["semantic"] = "semantic"
    breakpoint_threshold_type: BreakpointType = BreakpointType.PERCENTILE
    breakpoint_threshold_amount: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description=(
            "Percentile (0-100), std-dev multiplier (1-3), or IQR multiplier (0.5-1.5) "
            "depending on breakpoint_threshold_type"
        ),
    )
    buffer_size: int = Field(1, ge=1, le=5, description="Adjacent sentences to group before comparing")
    min_chunk_size: int = Field(
        100,
        ge=50,
        description=(
            "Minimum tokens per chunk. CRITICAL: prevents tiny fragments "
            "that break end-to-end accuracy (FloTorch 2026 finding)"
        ),
    )


class ContextualChunkingConfig(StrictModel):
    """
    Anthropic Contextual Retrieval (2024).
    Prepends an LLM-generated context description to each chunk before embedding.
    Reduces retrieval failures by 49-67% depending on pipeline configuration.
    Cost: ~$1.02/M document tokens with prompt caching.
    """

    type: Literal["contextual"] = "contextual"
    chunk_size: int = Field(800, ge=64, le=8192)
    chunk_overlap: int = Field(80, ge=0)
    context_model: str = Field(
        "claude-3-haiku-20240307",
        description="LLM for context generation. Haiku recommended for cost efficiency.",
    )
    context_prompt: str = Field(
        default=(
            "<document>\n{{WHOLE_DOCUMENT}}\n</document>\n"
            "Here is the chunk we want to situate within the whole document\n"
            "<chunk>\n{{CHUNK_CONTENT}}\n</chunk>\n"
            "Please give a short succinct context to situate this chunk within the overall "
            "document for the purposes of improving search retrieval of the chunk. "
            "Answer only with the succinct context and nothing else."
        ),
        description="Prompt template for context generation. "
        "Must contain {{WHOLE_DOCUMENT}} and {{CHUNK_CONTENT}} placeholders.",
    )

    @field_validator("context_prompt")
    @classmethod
    def _prompt_has_placeholders(cls, v: str) -> str:
        if "{{WHOLE_DOCUMENT}}" not in v:
            raise ValueError("context_prompt must contain {{WHOLE_DOCUMENT}} placeholder")
        if "{{CHUNK_CONTENT}}" not in v:
            raise ValueError("context_prompt must contain {{CHUNK_CONTENT}} placeholder")
        return v

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> "ContextualChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class LateChunkingConfig(StrictModel):
    """
    Jina AI Late Chunking (2024).
    Embeds the full document first, then pools token embeddings per chunk span.
    Preserves cross-chunk context (pronouns, references).
    Requires Jina embedding models (jina-embeddings-v3).
    """

    type: Literal["late"] = "late"
    chunk_size: int = Field(
        256,
        ge=64,
        le=2048,
        description="Chunk span size after late pooling",
    )


class PageLevelChunkingConfig(StrictModel):
    """
    Page-level chunking — each PDF page is a chunk.
    Winner of NVIDIA 2024 benchmark (0.648 accuracy, lowest variance).
    Best for financial reports, legal documents, paginated PDFs.
    """

    type: Literal["page_level"] = "page_level"
    # No parameters — page boundaries are natural document units


class PropositionChunkingConfig(StrictModel):
    """
    Dense-X Retrieval (Chen et al., 2023).
    Decomposes paragraphs into atomic, self-contained propositions.
    +17-25% Recall@5 improvement on entity-centric queries.
    Cost: one LLM call per paragraph.
    """

    type: Literal["proposition"] = "proposition"
    extraction_model: str = Field(
        "gpt-4o-mini",
        description="LLM for proposition extraction. gpt-4o for max quality, mini for cost.",
    )


ChunkingConfig = Annotated[
    Union[
        FixedChunkingConfig,
        RecursiveChunkingConfig,
        SemanticChunkingConfig,
        ContextualChunkingConfig,
        LateChunkingConfig,
        PageLevelChunkingConfig,
        PropositionChunkingConfig,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIEmbeddingConfig(StrictModel):
    type: Literal["openai"] = "openai"
    model: Literal[
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ] = "text-embedding-3-small"
    dimensions: int | None = Field(
        None,
        description=(
            "Matryoshka dimension reduction. None = model default. "
            "3-large supports any dim ≥ 256; 3-small ≥ 512."
        ),
    )
    # API key from env: OPENAI_API_KEY


class CohereEmbeddingConfig(StrictModel):
    type: Literal["cohere"] = "cohere"
    model: Literal[
        "embed-v4.0",
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
    ] = "embed-v4.0"
    input_type: Literal["search_document", "search_query"] = "search_document"
    # API key from env: COHERE_API_KEY


class VoyageEmbeddingConfig(StrictModel):
    type: Literal["voyage"] = "voyage"
    model: Literal[
        "voyage-3-large",
        "voyage-3",
        "voyage-3-lite",
        "voyage-code-3",
        "voyage-finance-2",
        "voyage-law-2",
    ] = "voyage-3-large"
    # API key from env: VOYAGE_API_KEY


class GeminiEmbeddingConfig(StrictModel):
    type: Literal["gemini"] = "gemini"
    model: Literal["text-embedding-004", "embedding-001"] = "text-embedding-004"
    # API key from env: GOOGLE_API_KEY


class BGEM3EmbeddingConfig(StrictModel):
    """
    BGE-M3 (BAAI, 2024) — unique triple capability: dense + sparse + multi-vector.
    Self-hosted only. MTEB: 63.0. Best open-source choice for hybrid search.
    Requires: FlagEmbedding>=1.2.0, torch>=2.0.0
    """

    type: Literal["bge_m3"] = "bge_m3"
    use_fp16: bool = Field(True, description="Half-precision inference (faster, minimal quality loss)")
    batch_size: int = Field(32, ge=1, le=256)
    # Self-hosted — no API key required


class NomicEmbeddingConfig(StrictModel):
    """
    Nomic embed-text-v1.5 — fully open (weights + training data + code).
    Matryoshka support: 64, 128, 256, 512, 768 dimensions.
    """

    type: Literal["nomic"] = "nomic"
    model: Literal["nomic-embed-text-v1.5", "nomic-embed-text-v1"] = "nomic-embed-text-v1.5"
    dimensionality: int = Field(
        768,
        description="Matryoshka reduction target. Supported: 64, 128, 256, 512, 768",
    )

    @field_validator("dimensionality")
    @classmethod
    def _valid_matryoshka_dim(cls, v: int) -> int:
        valid = {64, 128, 256, 512, 768}
        if v not in valid:
            raise ValueError(f"dimensionality must be one of {sorted(valid)}, got {v}")
        return v


class JinaEmbeddingConfig(StrictModel):
    """
    Jina v3 — required for late chunking. Task-specific LoRA adapters.
    late_chunking=True requires jina-embeddings-v3 (8K context window).
    """

    type: Literal["jina"] = "jina"
    model: Literal[
        "jina-embeddings-v3",
        "jina-embeddings-v2-base-en",
    ] = "jina-embeddings-v3"
    late_chunking: bool = Field(
        False,
        description=(
            "Enable Jina late chunking. Must be True when chunking.type='late'. "
            "Requires jina-embeddings-v3."
        ),
    )
    # API key from env: JINA_API_KEY


EmbeddingConfig = Annotated[
    Union[
        OpenAIEmbeddingConfig,
        CohereEmbeddingConfig,
        VoyageEmbeddingConfig,
        GeminiEmbeddingConfig,
        BGEM3EmbeddingConfig,
        NomicEmbeddingConfig,
        JinaEmbeddingConfig,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Vector Database
# ─────────────────────────────────────────────────────────────────────────────


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class ChromaDBConfig(StrictModel):
    """Local embedded vector store. Prototyping only — max ~7M vectors tested."""

    type: Literal["chromadb"] = "chromadb"
    collection_name: str = "ragfactory"
    persist_directory: str = Field(".chroma", description="Local directory for persistence")
    distance_metric: DistanceMetric = DistanceMetric.COSINE


class QdrantConfig(StrictModel):
    """
    Qdrant (Rust) — production recommendation.
    Benchmark leader: 8,500–12,000 QPS at 98.5% recall.
    Best-in-class metadata filtering.
    """

    type: Literal["qdrant"] = "qdrant"
    collection_name: str = "ragfactory"
    url: str = Field("http://localhost:6333", description="Qdrant server URL")
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    # API key from env: QDRANT_API_KEY (optional for local, required for Qdrant Cloud)


class PineconeConfig(StrictModel):
    """Managed serverless — up to 5,700 QPS at 1.4B vectors. Zero-ops."""

    type: Literal["pinecone"] = "pinecone"
    index_name: str = "ragfactory"
    environment: str = Field("us-east-1-aws", description="Pinecone cloud region")
    # API key from env: PINECONE_API_KEY


class WeaviateConfig(StrictModel):
    """Weaviate — modular, GraphQL API, native hybrid search, multi-tenancy."""

    type: Literal["weaviate"] = "weaviate"
    class_name: str = Field("RagFactory", description="Weaviate class name (PascalCase)")
    url: str = Field("http://localhost:8080")
    # API key from env: WEAVIATE_API_KEY (optional for local)


class MilvusConfig(StrictModel):
    """Milvus — distributed, GPU-accelerated (CAGRA), billion-scale."""

    type: Literal["milvus"] = "milvus"
    collection_name: str = "ragfactory"
    uri: str = Field("http://localhost:19530", description="Milvus server URI or Zilliz Cloud URL")
    # Token from env: MILVUS_TOKEN (required for Zilliz Cloud)


class PgVectorConfig(StrictModel):
    """pgvector — PostgreSQL extension. Best for teams already on Postgres."""

    type: Literal["pgvector"] = "pgvector"
    collection_name: str = "ragfactory"
    # Connection string from env: PGVECTOR_CONNECTION_STRING


VectorDBConfig = Annotated[
    Union[
        ChromaDBConfig,
        QdrantConfig,
        PineconeConfig,
        WeaviateConfig,
        MilvusConfig,
        PgVectorConfig,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Indexing  (chunking + embedding + vectordb as a unit)
# ─────────────────────────────────────────────────────────────────────────────


class IndexingConfig(StrictModel):
    chunking: ChunkingConfig = Field(default_factory=RecursiveChunkingConfig)
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Retrieval
# ─────────────────────────────────────────────────────────────────────────────


class QueryRewritingStrategy(str, Enum):
    MULTI_QUERY = "multi_query"
    SUB_QUESTION = "sub_question"
    STEP_BACK = "step_back"


class QueryRewritingConfig(StrictModel):
    """
    Multi-query rewriting: generates N rephrasings → retrieves across all → deduplicates.
    +25-50% precision improvement on complex queries.
    """

    enabled: bool = True
    strategy: QueryRewritingStrategy = QueryRewritingStrategy.MULTI_QUERY
    num_rewrites: int = Field(3, ge=1, le=10, description="Sweet spot: 3-5 variants")
    rewrite_model: str | None = Field(
        None,
        description=(
            "Override LLM for rewriting. Set to a cheaper model (e.g. gpt-4o-mini). "
            "None = inherit from generation.llm"
        ),
    )


class HyDEConfig(StrictModel):
    """
    Hypothetical Document Embeddings (Gao et al., 2022).
    Generates a hypothetical answer, embeds that instead of the raw query.
    Best for short/ambiguous queries and vocabulary-gap domains.
    Adds ~1 LLM call latency per query.
    """

    enabled: bool = True
    num_hypotheses: int = Field(1, ge=1, le=5)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class RouteDefinition(StrictModel):
    name: str = Field(..., description="Route identifier (used in generated code)")
    description: str = Field(
        ...,
        description="Natural language description — this is what the router LLM reads",
    )


class RoutingConfig(StrictModel):
    enabled: bool = True
    routes: list[RouteDefinition] = Field(
        ...,
        min_length=2,
        description="At least 2 routes required for routing to be meaningful",
    )
    confidence_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to route. Below threshold → fallback_route",
    )
    fallback_route: str | None = Field(
        None,
        description="Route name for low-confidence queries. None = first route",
    )


class PreRetrievalConfig(StrictModel):
    query_rewriting: QueryRewritingConfig | None = None
    hyde: HyDEConfig | None = None
    routing: RoutingConfig | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────


class DenseRetrievalConfig(StrictModel):
    """Pure vector similarity search. Fast, simple. Baseline for all comparisons."""

    type: Literal["dense"] = "dense"
    top_k: int = Field(20, ge=1, le=500)


class HybridRRFConfig(StrictModel):
    """
    Hybrid search with Reciprocal Rank Fusion.
    Runs BM25 (sparse) + dense in parallel, fuses with RRF.
    +15-30% recall vs either method alone (IBM Research, 2024).
    Recommended default for production.
    """

    type: Literal["hybrid_rrf"] = "hybrid_rrf"
    top_k: int = Field(100, ge=1, le=500, description="Candidates per retriever before fusion")
    rrf_k: int = Field(
        60,
        ge=1,
        description="RRF smoothing constant. k=60 is well-validated default.",
    )
    bm25_k1: float = Field(
        1.2,
        ge=0.0,
        le=3.0,
        description="BM25 term frequency saturation. Typical range: 1.2-2.0",
    )
    bm25_b: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="BM25 document length normalization. 0=none, 1=full",
    )


class HybridWeightedConfig(StrictModel):
    """
    Hybrid search with weighted linear combination.
    More control than RRF but requires alpha tuning per domain.
    alpha=1.0 → pure dense; alpha=0.0 → pure sparse.
    """

    type: Literal["hybrid_weighted"] = "hybrid_weighted"
    top_k: int = Field(100, ge=1, le=500)
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Dense weight. Sparse weight = (1 - alpha).",
    )
    bm25_k1: float = Field(1.2, ge=0.0, le=3.0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)


class SmallToBigConfig(StrictModel):
    """
    Parent-Document Retrieval (LlamaIndex / LangChain).
    Retrieve small child chunks for precision, return larger parent chunks for context.
    Solves embedding dilution on dense documents.
    """

    type: Literal["small_to_big"] = "small_to_big"
    child_chunk_size: int = Field(256, ge=64, le=1024)
    parent_chunk_size: int = Field(1024, ge=256, le=8192)
    top_k: int = Field(5, ge=1, le=50)

    @model_validator(mode="after")
    def _parent_larger_than_child(self) -> "SmallToBigConfig":
        if self.parent_chunk_size <= self.child_chunk_size:
            raise ValueError(
                f"parent_chunk_size ({self.parent_chunk_size}) must be greater than "
                f"child_chunk_size ({self.child_chunk_size})"
            )
        return self


class SentenceWindowConfig(StrictModel):
    """
    Sentence Window Retrieval (LlamaIndex).
    Match individual sentences (precision), expand context window for synthesis.
    Default window_size=5 retrieves ±5 sentences around each match.
    """

    type: Literal["sentence_window"] = "sentence_window"
    window_size: int = Field(5, ge=1, le=15, description="Sentences on each side of matched sentence")
    top_k: int = Field(5, ge=1, le=50)


RetrievalConfig = Annotated[
    Union[
        DenseRetrievalConfig,
        HybridRRFConfig,
        HybridWeightedConfig,
        SmallToBigConfig,
        SentenceWindowConfig,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Post-Retrieval
# ─────────────────────────────────────────────────────────────────────────────


class CohereRerankerConfig(StrictModel):
    """Cohere Rerank API — high accuracy, fast, 100+ languages, no GPU needed."""

    type: Literal["cohere"] = "cohere"
    model: Literal[
        "rerank-v3.5",
        "rerank-english-v3.0",
        "rerank-multilingual-v3.0",
    ] = "rerank-v3.5"
    top_n: int = Field(5, ge=1, le=50, description="Final documents after reranking")
    # API key from env: COHERE_API_KEY


class CrossEncoderRerankerConfig(StrictModel):
    """Cross-encoder reranker. Best quality, requires GPU for production throughput."""

    type: Literal["cross_encoder"] = "cross_encoder"
    model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="HuggingFace model ID. Alternatives: BAAI/bge-reranker-large",
    )
    top_n: int = Field(5, ge=1, le=50)


class ColBERTRerankerConfig(StrictModel):
    """
    ColBERT late-interaction reranker via RAGatouille.
    Per-token MaxSim scoring — better than bi-encoder, faster than cross-encoder.
    ColBERTv2: 6-10x storage compression over v1.
    """

    type: Literal["colbert"] = "colbert"
    model: str = Field("colbert-ir/colbertv2.0")
    top_n: int = Field(5, ge=1, le=50)


class FlashRankRerankerConfig(StrictModel):
    """
    FlashRank — CPU-only, minimal dependencies (~4MB footprint).
    Best for cost-sensitive or edge deployments where GPU is unavailable.
    """

    type: Literal["flashrank"] = "flashrank"
    model: Literal[
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L-12-v2",
        "rank-T5-flan",
        "ms-marco-MultiBERT-L-12",
    ] = "ms-marco-MiniLM-L-12-v2"
    top_n: int = Field(5, ge=1, le=50)


RerankerConfig = Annotated[
    Union[
        CohereRerankerConfig,
        CrossEncoderRerankerConfig,
        ColBERTRerankerConfig,
        FlashRankRerankerConfig,
    ],
    Field(discriminator="type"),
]


class CompressionMethod(str, Enum):
    EXTRACTIVE = "extractive"
    LLM_LINGUA = "llm_lingua"
    SENTENCE_FILTER = "sentence_filter"


class ContextCompressionConfig(StrictModel):
    """
    Reduce retrieved context before passing to LLM.
    Mitigates the "lost-in-the-middle" effect (>30% degradation beyond ~10 docs).
    """

    enabled: bool = True
    method: CompressionMethod = CompressionMethod.EXTRACTIVE
    compression_ratio: float = Field(0.5, ge=0.1, le=0.9)
    max_context_tokens: int = Field(4096, ge=512)


class ContextOrderingStrategy(str, Enum):
    RELEVANCE_FIRST = "relevance_first"
    CHRONOLOGICAL = "chronological"
    REVERSE_RELEVANCE = "reverse_relevance"


class ContextAssemblyConfig(StrictModel):
    ordering: ContextOrderingStrategy = ContextOrderingStrategy.RELEVANCE_FIRST
    max_sources: int = Field(5, ge=1, le=20)
    source_attribution: bool = True
    context_separator: str = "\n---\n"


class PostRetrievalConfig(StrictModel):
    reranker: RerankerConfig | None = None
    context_compression: ContextCompressionConfig | None = None
    context_assembly: ContextAssemblyConfig = Field(default_factory=ContextAssemblyConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Generation — LLM
# ─────────────────────────────────────────────────────────────────────────────


class OpenAILLMConfig(StrictModel):
    type: Literal["openai"] = "openai"
    model: Literal["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"] = "gpt-4o-mini"
    temperature: float = Field(0.05, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=128, le=16384)
    # API key from env: OPENAI_API_KEY


class AnthropicLLMConfig(StrictModel):
    type: Literal["anthropic"] = "anthropic"
    model: Literal[
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ] = "claude-sonnet-4-6"
    temperature: float = Field(0.05, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=128, le=8192)
    # API key from env: ANTHROPIC_API_KEY


class CohereLLMConfig(StrictModel):
    type: Literal["cohere_llm"] = "cohere_llm"
    model: Literal["command-r-plus", "command-r", "command"] = "command-r"
    temperature: float = Field(0.05, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=128, le=4096)
    # API key from env: COHERE_API_KEY


class OllamaLLMConfig(StrictModel):
    """Local inference via Ollama. Zero API cost. Requires Ollama running locally."""

    type: Literal["ollama"] = "ollama"
    model: str = Field("llama3.2", description="Model name as shown in `ollama list`")
    base_url: str = Field("http://localhost:11434")
    temperature: float = Field(0.05, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=128)


LLMConfig = Annotated[
    Union[
        OpenAILLMConfig,
        AnthropicLLMConfig,
        CohereLLMConfig,
        OllamaLLMConfig,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Generation — Advanced (schema only in this release)
# ─────────────────────────────────────────────────────────────────────────────


class CRAGConfig(StrictModel):
    """
    Corrective RAG (Yan et al., 2024 — arXiv:2401.15884).
    Evaluates retrieval quality and triggers corrective actions:
      - Correct  (score > upper_threshold): refine retrieved docs
      - Ambiguous (lower < score < upper):  combine refined + web search
      - Incorrect (score < lower_threshold): use web search only
    Evaluator accuracy: 84.3% vs ChatGPT's 58-64.7%.
    Note: web_search_fallback=True adds latency. Disable in air-gapped environments.
    """

    enabled: bool = True
    upper_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Confidence above this → Correct action (refine docs)",
    )
    lower_threshold: float = Field(
        -0.5,
        le=0.0,
        description="Confidence below this → Incorrect action (web search fallback)",
    )
    web_search_fallback: bool = Field(
        True,
        description="Trigger web search when retrieval is low-confidence. Adds ~1-3s latency.",
    )
    web_search_provider: Literal["tavily", "serper", "duckduckgo"] = "tavily"
    # Parsed for future CRAG support; generation rejects enabled CRAG in this release.


class FLAREConfig(StrictModel):
    """
    FLARE — Forward-Looking Active Retrieval (Jiang et al., EMNLP 2023).
    Monitors token-level generation confidence. Triggers retrieval when
    any token probability drops below threshold.
    IMPORTANT: Requires a model that exposes logprobs. Anthropic API does NOT
    support logprobs — use OpenAI or open-source models.
    """

    enabled: bool = True
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_iterations: int = Field(5, ge=1, le=20)


class AgenticConfig(StrictModel):
    """
    Agentic RAG — LLM agent with reflection, planning, and tool use.
    Complex query accuracy: 34% → 78% (NVIDIA, 2025).
    Cost: multiple LLM calls per query. Use for complex, multi-step tasks only.
    """

    enabled: bool = True
    max_reasoning_steps: int = Field(5, ge=1, le=20)
    tools: list[Literal["web_search", "calculator", "code_interpreter"]] = Field(
        default_factory=list
    )
    reflection_enabled: bool = Field(
        True,
        description="Agent evaluates retrieval quality before generating",
    )


class AdvancedGenerationConfig(StrictModel):
    """Advanced generation options parsed by the schema but blocked by generation."""

    crag: CRAGConfig | None = None
    flare: FLAREConfig | None = None
    agentic: AgenticConfig | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Generation (combined)
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the provided context.\n"
    "If the answer cannot be found in the context, say: "
    '"I cannot find this information in the provided documents."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class GenerationConfig(StrictModel):
    llm: LLMConfig
    prompt_template: str = Field(
        default=_DEFAULT_PROMPT,
        description=(
            "RAG system prompt template. "
            "Must contain {context} and {question} placeholders."
        ),
    )
    advanced: AdvancedGenerationConfig | None = None

    @field_validator("prompt_template")
    @classmethod
    def _template_has_placeholders(cls, v: str) -> str:
        missing = []
        if "{context}" not in v:
            missing.append("{context}")
        if "{question}" not in v:
            missing.append("{question}")
        if missing:
            raise ValueError(
                f"prompt_template missing required placeholders: {missing}"
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


class EvalFramework(str, Enum):
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"
    BOTH = "both"


class EvaluationConfig(StrictModel):
    framework: EvalFramework = EvalFramework.RAGAS
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ],
        min_length=1,
    )
    judge_model: str = Field("gpt-4o", description="LLM for evaluation judge calls")
    num_test_cases: int = Field(50, ge=5, le=1000)
    pass_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score across all metrics to pass evaluation gate",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────────────────


class RAGPipelineConfig(StrictModel):
    """
    Complete RAGFactory pipeline configuration.

    Serialize to YAML for storage and version control.
    Deserialize to generate production-ready pipeline code.

    Example:
        config = RAGPipelineConfig(
            name="my-legal-rag",
            framework=Framework.LANGCHAIN,
            indexing=IndexingConfig(
                embedding=VoyageEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            retrieval=HybridRRFConfig(),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        )
        yaml_str = config.to_yaml()
        loaded = RAGPipelineConfig.from_yaml(yaml_str)
        assert config == loaded
    """

    version: Literal["1.0"] = "1.0"
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9\-]*$",
        description="Pipeline name. Lowercase letters, numbers, and hyphens only.",
    )
    framework: Framework = Framework.LANGCHAIN
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    indexing: IndexingConfig
    pre_retrieval: PreRetrievalConfig = Field(default_factory=PreRetrievalConfig)
    retrieval: RetrievalConfig = Field(default_factory=HybridRRFConfig)
    post_retrieval: PostRetrievalConfig = Field(default_factory=PostRetrievalConfig)
    generation: GenerationConfig
    evaluation: EvaluationConfig | None = None

    # ── Serialization ────────────────────────────────────────────────────────

    def to_yaml(self) -> str:
        """Serialize to YAML string. Safe for git commit — no secrets included."""
        return yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def to_yaml_file(self, path: str) -> None:
        """Write config to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "RAGPipelineConfig":
        """Deserialize from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "RAGPipelineConfig":
        """Load from a YAML file path."""
        with open(path, encoding="utf-8") as f:
            return cls.from_yaml(f.read())
