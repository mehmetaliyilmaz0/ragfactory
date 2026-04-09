"""
Dependency manifest for RAGFactory generated pipelines.

Design principles:
  - NO auto-resolve. Every version range here has been manually tested.
  - Ranges are intentionally conservative: pin the major, allow minor/patch bumps.
  - When a framework releases a new major, this file is updated + regression-tested
    before the new version is accepted.
  - get_dependencies() is the single public API. Returns a deduplicated, sorted list
    of pip requirement strings for a given pipeline config.

Structure:
  DEPENDENCY_MATRIX[framework][category][component_type] → list[str]

Categories:
  base        — always included for the framework
  embedding   — keyed by EmbeddingConfig.type
  vectordb    — keyed by VectorDBConfig.type
  reranker    — keyed by RerankerConfig.type
  llm         — keyed by LLMConfig.type
  chunking    — keyed by ChunkingConfig.type (only entries that have extra deps)
  parser      — keyed by ParserType value
  evaluation  — keyed by EvalFramework value
"""

from __future__ import annotations

# ─── Dependency Matrix ────────────────────────────────────────────────────────
# Each leaf value is a list of pip-installable requirement strings.
# Use the most restrictive range that is known to work.

DEPENDENCY_MATRIX: dict[str, dict[str, dict[str, list[str]]]] = {
    # ──────────────────────────────────────────────────────────────────────────
    # LangChain
    # ──────────────────────────────────────────────────────────────────────────
    "langchain": {
        "base": {
            "_": [
                "langchain>=0.3.0,<0.4.0",
                "langchain-core>=0.3.0,<0.4.0",
                "langchain-community>=0.3.0,<0.4.0",
                "langchain-text-splitters>=0.3.0,<0.4.0",
            ]
        },
        "embedding": {
            "openai": [
                "langchain-openai>=0.2.0,<0.3.0",
                "openai>=1.40.0,<2.0.0",
            ],
            "cohere": [
                "langchain-cohere>=0.3.0,<0.4.0",
                "cohere>=5.0.0,<6.0.0",
            ],
            "voyage": [
                "voyageai>=0.3.0,<1.0.0",
            ],
            "gemini": [
                "langchain-google-genai>=2.0.0,<3.0.0",
                "google-generativeai>=0.7.0,<1.0.0",
            ],
            "bge_m3": [
                "FlagEmbedding>=1.2.0,<2.0.0",
                "torch>=2.0.0",
            ],
            "nomic": [
                "nomic>=3.0.0,<4.0.0",
            ],
            "jina": [
                "langchain-community>=0.3.0,<0.4.0",  # uses JinaEmbeddings via community
                "requests>=2.31.0",
            ],
        },
        "vectordb": {
            "chromadb": [
                "langchain-chroma>=0.1.0,<0.2.0",
                "chromadb>=0.5.0,<1.0.0",
            ],
            "qdrant": [
                "langchain-qdrant>=0.2.0,<0.3.0",
                "qdrant-client>=1.7.0,<2.0.0",
            ],
            "pinecone": [
                "langchain-pinecone>=0.2.0,<0.3.0",
                "pinecone-client>=3.0.0,<4.0.0",
            ],
            "weaviate": [
                "langchain-weaviate>=0.0.3,<0.1.0",
                "weaviate-client>=4.0.0,<5.0.0",
            ],
            "milvus": [
                "langchain-milvus>=0.1.0,<0.2.0",
                "pymilvus>=2.4.0,<3.0.0",
            ],
            "pgvector": [
                "langchain-postgres>=0.0.9,<0.1.0",
                "psycopg[binary]>=3.1.0,<4.0.0",
                "pgvector>=0.2.0",
            ],
        },
        "reranker": {
            "cohere": [
                "cohere>=5.0.0,<6.0.0",
            ],
            "cross_encoder": [
                "sentence-transformers>=3.0.0,<4.0.0",
                "torch>=2.0.0",
            ],
            "colbert": [
                "ragatouille>=0.0.8,<0.1.0",
            ],
            "flashrank": [
                "flashrank>=0.2.0,<1.0.0",
            ],
        },
        "llm": {
            "openai": [
                "langchain-openai>=0.2.0,<0.3.0",
                "openai>=1.40.0,<2.0.0",
            ],
            "anthropic": [
                "langchain-anthropic>=0.3.0,<0.4.0",
                "anthropic>=0.34.0,<1.0.0",
            ],
            "cohere_llm": [
                "langchain-cohere>=0.3.0,<0.4.0",
                "cohere>=5.0.0,<6.0.0",
            ],
            "ollama": [
                "langchain-ollama>=0.2.0,<0.3.0",
            ],
        },
    },

    # ──────────────────────────────────────────────────────────────────────────
    # LlamaIndex
    # ──────────────────────────────────────────────────────────────────────────
    "llamaindex": {
        "base": {
            "_": [
                "llama-index>=0.11.0,<0.12.0",
                "llama-index-core>=0.11.0,<0.12.0",
            ]
        },
        "embedding": {
            "openai": [
                "llama-index-embeddings-openai>=0.2.0,<0.3.0",
                "openai>=1.40.0,<2.0.0",
            ],
            "cohere": [
                "llama-index-embeddings-cohere>=0.2.0,<0.3.0",
                "cohere>=5.0.0,<6.0.0",
            ],
            "voyage": [
                "llama-index-embeddings-voyageai>=0.2.0,<0.3.0",
                "voyageai>=0.3.0,<1.0.0",
            ],
            "gemini": [
                "llama-index-embeddings-gemini>=0.2.0,<0.3.0",
                "google-generativeai>=0.7.0,<1.0.0",
            ],
            "bge_m3": [
                "llama-index-embeddings-huggingface>=0.3.0,<0.4.0",
                "FlagEmbedding>=1.2.0,<2.0.0",
                "torch>=2.0.0",
            ],
            "nomic": [
                "llama-index-embeddings-nomic>=0.2.0,<0.3.0",
                "nomic>=3.0.0,<4.0.0",
            ],
            "jina": [
                "llama-index-embeddings-jinaai>=0.2.0,<0.3.0",
            ],
        },
        "vectordb": {
            "chromadb": [
                "llama-index-vector-stores-chroma>=0.2.0,<0.3.0",
                "chromadb>=0.5.0,<1.0.0",
            ],
            "qdrant": [
                "llama-index-vector-stores-qdrant>=0.3.0,<0.4.0",
                "qdrant-client>=1.7.0,<2.0.0",
            ],
            "pinecone": [
                "llama-index-vector-stores-pinecone>=0.2.0,<0.3.0",
                "pinecone-client>=3.0.0,<4.0.0",
            ],
            "weaviate": [
                "llama-index-vector-stores-weaviate>=0.2.0,<0.3.0",
                "weaviate-client>=4.0.0,<5.0.0",
            ],
            "milvus": [
                "llama-index-vector-stores-milvus>=0.2.0,<0.3.0",
                "pymilvus>=2.4.0,<3.0.0",
            ],
            "pgvector": [
                "llama-index-vector-stores-postgres>=0.2.0,<0.3.0",
                "psycopg[binary]>=3.1.0,<4.0.0",
                "pgvector>=0.2.0",
            ],
        },
        "reranker": {
            "cohere": [
                "llama-index-postprocessor-cohere-rerank>=0.2.0,<0.3.0",
                "cohere>=5.0.0,<6.0.0",
            ],
            "cross_encoder": [
                "llama-index-postprocessor-sbert-rerank>=0.2.0,<0.3.0",
                "sentence-transformers>=3.0.0,<4.0.0",
                "torch>=2.0.0",
            ],
            "colbert": [
                "ragatouille>=0.0.8,<0.1.0",
            ],
            "flashrank": [
                "llama-index-postprocessor-flashrank-reranker>=0.2.0,<0.3.0",
            ],
        },
        "llm": {
            "openai": [
                "llama-index-llms-openai>=0.2.0,<0.3.0",
                "openai>=1.40.0,<2.0.0",
            ],
            "anthropic": [
                "llama-index-llms-anthropic>=0.4.0,<0.5.0",
                "anthropic>=0.34.0,<1.0.0",
            ],
            "cohere_llm": [
                "llama-index-llms-cohere>=0.2.0,<0.3.0",
                "cohere>=5.0.0,<6.0.0",
            ],
            "ollama": [
                "llama-index-llms-ollama>=0.3.0,<0.4.0",
            ],
        },
    },
}

# ─── Chunking Extra Dependencies ──────────────────────────────────────────────
# These are framework-independent — same package regardless of langchain/llamaindex.

CHUNKING_EXTRA_DEPS: dict[str, list[str]] = {
    "fixed": [],        # no extra deps — built into both frameworks
    "recursive": [],    # no extra deps — built into both frameworks
    "semantic": [
        "sentence-transformers>=3.0.0,<4.0.0",
    ],
    "contextual": [
        "anthropic>=0.34.0,<1.0.0",  # default context model is claude-haiku
    ],
    "late": [
        # Late chunking requires Jina embedding (enforced by validator)
        # No extra chunking-specific deps beyond Jina embedding deps
    ],
    "page_level": [],   # PDF page splitting — handled by parser
    "proposition": [
        "openai>=1.40.0,<2.0.0",  # default extraction model is gpt-4o-mini
    ],
}

# ─── Parser Extra Dependencies ────────────────────────────────────────────────

PARSER_EXTRA_DEPS: dict[str, list[str]] = {
    "default": [],
    "unstructured": [
        "unstructured[pdf,docx,pptx]>=0.15.0,<1.0.0",
    ],
    "azure_doc_intelligence": [
        "azure-ai-documentintelligence>=1.0.0,<2.0.0",
    ],
    "docling": [
        "docling>=2.0.0,<3.0.0",
    ],
}

# ─── Evaluation Extra Dependencies ────────────────────────────────────────────

EVALUATION_EXTRA_DEPS: dict[str, list[str]] = {
    "ragas": [
        "ragas>=0.1.21,<0.2.0",
        "datasets>=2.14.0",
    ],
    "deepeval": [
        "deepeval>=1.0.0,<2.0.0",
    ],
    "both": [
        "ragas>=0.1.21,<0.2.0",
        "datasets>=2.14.0",
        "deepeval>=1.0.0,<2.0.0",
    ],
}

# ─── Web Search (CRAG) ────────────────────────────────────────────────────────

WEB_SEARCH_DEPS: dict[str, list[str]] = {
    "tavily": ["tavily-python>=0.3.0,<1.0.0"],
    "serper": ["google-search-results>=2.4.0"],
    "duckduckgo": ["duckduckgo-search>=6.0.0,<7.0.0"],
}


# ─── Public API ───────────────────────────────────────────────────────────────


def get_dependencies(config: object) -> list[str]:  # noqa: ANN001
    """
    Return a deduplicated, sorted list of pip requirement strings for the given config.

    This is the single entry point used by the code generator when producing
    pyproject.toml / requirements.txt for the generated pipeline.

    Args:
        config: A RAGPipelineConfig instance. Typed as object to avoid a circular
                import (config.py → versions.py → config.py).

    Returns:
        Sorted list of unique requirement strings, e.g.:
        ["anthropic>=0.34.0,<1.0.0", "langchain>=0.3.0,<0.4.0", ...]
    """
    # Avoid circular import — RAGPipelineConfig is only imported at call time.
    from ragfactory.core.config import RAGPipelineConfig  # noqa: PLC0415

    if not isinstance(config, RAGPipelineConfig):
        raise TypeError(f"Expected RAGPipelineConfig, got {type(config).__name__}")

    deps: set[str] = set()
    framework = config.framework  # str value via use_enum_values=True
    matrix = DEPENDENCY_MATRIX[framework]

    # ── Base framework packages ────────────────────────────────────────────
    deps.update(matrix["base"]["_"])

    # ── Embedding ─────────────────────────────────────────────────────────
    emb_type = config.indexing.embedding.type
    if emb_type in matrix["embedding"]:
        deps.update(matrix["embedding"][emb_type])

    # ── Vector DB ─────────────────────────────────────────────────────────
    vdb_type = config.indexing.vector_db.type
    if vdb_type in matrix["vectordb"]:
        deps.update(matrix["vectordb"][vdb_type])

    # ── Reranker (optional) ───────────────────────────────────────────────
    if config.post_retrieval.reranker is not None:
        reranker_type = config.post_retrieval.reranker.type
        if reranker_type in matrix["reranker"]:
            deps.update(matrix["reranker"][reranker_type])

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_type = config.generation.llm.type
    if llm_type in matrix["llm"]:
        deps.update(matrix["llm"][llm_type])

    # ── Chunking extras ───────────────────────────────────────────────────
    chunking_type = config.indexing.chunking.type
    deps.update(CHUNKING_EXTRA_DEPS.get(chunking_type, []))

    # ── Parser extras ─────────────────────────────────────────────────────
    parser_value = config.ingestion.parser  # str value
    deps.update(PARSER_EXTRA_DEPS.get(parser_value, []))

    # ── Evaluation extras (optional) ──────────────────────────────────────
    if config.evaluation is not None:
        eval_framework = config.evaluation.framework  # str value
        deps.update(EVALUATION_EXTRA_DEPS.get(eval_framework, []))

    # ── CRAG web search extras ────────────────────────────────────────────
    if (
        config.generation.advanced is not None
        and config.generation.advanced.crag is not None
        and config.generation.advanced.crag.enabled
    ):
        provider = config.generation.advanced.crag.web_search_provider
        deps.update(WEB_SEARCH_DEPS.get(provider, []))

    return sorted(deps)
