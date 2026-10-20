"""
Dependency manifest for RAGFactory generated pipelines.

Design principles:
  - Explicit ranges, not runtime auto-resolution. Representative profiles are tested;
    optional combinations still need their own resolver and runtime checks.
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
                "langchain>=1.4,<2",
                "langchain-core>=1,<2",
                "langchain-text-splitters>=1,<2",
                "fastapi>=0.110.0",
                "uvicorn>=0.28.0,<1.0.0",
                "python-dotenv>=1,<2",
            ]
        },
        "embedding": {
            "openai": [
                "langchain-openai>=1,<2",
            ],
            "cohere": [
                "langchain-cohere>=0.6,<0.7",
            ],
            "voyage": [
                "langchain-voyageai>=0.4,<0.5",
            ],
            "gemini": [
                "langchain-google-genai>=4,<5",
            ],
            "bge_m3": [
                "langchain-huggingface>=1.2,<2",
                "sentence-transformers>=3,<6",
                "torch>=2.0.0",
            ],
            "nomic": [
                "langchain-nomic>=0.3,<1",
            ],
            "jina": [
                "langchain-community>=0.4,<0.5",
            ],
        },
        "vectordb": {
            "chromadb": [
                "langchain-chroma>=1,<2",
            ],
            "qdrant": [
                "langchain-qdrant>=1,<2",
            ],
            "pinecone": [
                "langchain-pinecone>=0.2,<1",
            ],
            "weaviate": [
                "langchain-weaviate>=0.0.8,<0.1",
            ],
            "milvus": [
                "langchain-milvus>=0.4,<0.5",
            ],
            "pgvector": [
                "langchain-postgres>=0.0.15,<0.1",
            ],
        },
        "reranker": {
            "cohere": [
                "langchain-cohere>=0.6,<0.7",
            ],
            "cross_encoder": [
                "langchain-community>=0.4,<0.5",
                "sentence-transformers>=3,<6",
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
                "langchain-openai>=1,<2",
            ],
            "anthropic": [
                "langchain-anthropic>=1,<2",
            ],
            "cohere_llm": [
                "langchain-cohere>=0.6,<0.7",
            ],
            "ollama": [
                "langchain-ollama>=1,<2",
            ],
        },
        "retrieval": {
            "hybrid_rrf": [
                "langchain-classic>=1,<2",
                "langchain-community>=0.4,<0.5",
                "rank-bm25>=0.2,<0.3",
            ],
            "hybrid_weighted": [
                "langchain-classic>=1,<2",
                "langchain-community>=0.4,<0.5",
                "rank-bm25>=0.2,<0.3",
            ],
            "small_to_big": ["langchain-classic>=1,<2"],
            "sentence_window": ["langchain-classic>=1,<2"],
        },
    },
    # ──────────────────────────────────────────────────────────────────────────
    # LlamaIndex
    # ──────────────────────────────────────────────────────────────────────────
    "llamaindex": {
        "base": {
            "_": [
                "llama-index-core>=0.14,<0.15",
                "fastapi>=0.110.0",
                "uvicorn>=0.28.0,<1.0.0",
                "python-dotenv>=1,<2",
            ]
        },
        "embedding": {
            "openai": [
                "llama-index-embeddings-openai>=0.7,<0.8",
                "openai>=1.40.0,<3.0.0",
            ],
            "cohere": [
                "llama-index-embeddings-cohere>=0.5,<1",
            ],
            "voyage": [
                "llama-index-embeddings-voyageai>=0.4,<1",
                "voyageai>=0.3,<2",
            ],
            "gemini": [
                "llama-index-embeddings-google-genai>=0.5,<0.6",
            ],
            "bge_m3": [
                "llama-index-embeddings-huggingface>=0.6,<1",
                "sentence-transformers>=3,<6",
            ],
            "nomic": [
                "llama-index-embeddings-nomic>=0.4,<1",
            ],
            "jina": [
                "llama-index-embeddings-jinaai>=0.4,<1",
            ],
        },
        "vectordb": {
            "chromadb": [
                "llama-index-vector-stores-chroma>=0.5,<1",
            ],
            "qdrant": [
                "llama-index-vector-stores-qdrant>=0.10,<0.11",
            ],
            "pinecone": [
                "llama-index-vector-stores-pinecone>=0.5,<1",
            ],
            "weaviate": [
                "llama-index-vector-stores-weaviate>=1,<2",
            ],
            "milvus": [
                "llama-index-vector-stores-milvus>=0.5,<1",
            ],
            "pgvector": [
                "llama-index-vector-stores-postgres>=0.5,<1",
            ],
        },
        "reranker": {
            "cohere": [
                "llama-index-postprocessor-cohere-rerank>=0.10,<0.11",
            ],
            "cross_encoder": [
                "llama-index-postprocessor-sbert-rerank>=0.5,<1",
                "sentence-transformers>=3,<6",
                "torch>=2.0.0",
            ],
            "colbert": [
                "ragatouille>=0.0.8,<0.1.0",
            ],
            "flashrank": [
                "llama-index-postprocessor-flashrank-reranker>=0.4,<1",
            ],
        },
        "llm": {
            "openai": [
                "llama-index-llms-openai>=0.8,<0.9",
                "openai>=1.40.0,<3.0.0",
            ],
            "anthropic": [
                "llama-index-llms-anthropic>=0.12,<0.13",
            ],
            "cohere_llm": [
                "llama-index-llms-cohere>=0.5,<1",
            ],
            "ollama": [
                "llama-index-llms-ollama>=0.11,<0.12",
            ],
        },
    },
}

# ─── Chunking Extra Dependencies ──────────────────────────────────────────────
# These are framework-independent — same package regardless of langchain/llamaindex.

CHUNKING_EXTRA_DEPS: dict[str, list[str]] = {
    "fixed": [],  # no extra deps — built into both frameworks
    "recursive": [],  # no extra deps — built into both frameworks
    "semantic": [],  # LangChain-only dependency is selected below.
    "contextual": [
        "anthropic>=0.34.0,<1.0.0",  # default context model is claude-haiku
    ],
    "late": [
        # Late chunking requires Jina embedding (enforced by validator)
        # No extra chunking-specific deps beyond Jina embedding deps
    ],
    "page_level": [],  # PDF page splitting — handled by parser
    "proposition": [
        "openai>=1.40.0,<3.0.0",  # default extraction model is gpt-4o-mini
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
        "ragas>=0.4,<0.5",
        # Ragas 0.4.3 still imports ChatVertexAI from the removed community module.
        "langchain-community>=0.4,<0.4.2",
        "openai>=1.40.0,<3.0.0",
    ],
    "deepeval": [
        "deepeval>=4,<5",
    ],
    "both": [
        "ragas>=0.4,<0.5",
        "langchain-community>=0.4,<0.4.2",
        "deepeval>=4,<5",
        "openai>=1.40.0,<3.0.0",
    ],
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

    # ── Agentic Flow packages ──────────────────────────────────────────────
    if config.flow_type == "agentic" and framework == "langchain":
        deps.add("langgraph>=1,<2")

    # ── Embedding ─────────────────────────────────────────────────────────
    emb_type = config.indexing.embedding.type
    if emb_type in matrix["embedding"]:
        deps.update(matrix["embedding"][emb_type])

    # ── Vector DB ─────────────────────────────────────────────────────────
    vdb_type = config.indexing.vector_db.type
    if vdb_type in matrix["vectordb"]:
        deps.update(matrix["vectordb"][vdb_type])
        if (
            framework == "llamaindex"
            and vdb_type == "qdrant"
            and config.retrieval.type in {"hybrid_rrf", "hybrid_weighted"}
        ):
            deps.add("llama-index-vector-stores-qdrant[fastembed]>=0.10,<0.11")

    # ── Reranker (optional) ───────────────────────────────────────────────
    if config.post_retrieval.reranker is not None:
        reranker_type = config.post_retrieval.reranker.type
        if reranker_type in matrix["reranker"]:
            deps.update(matrix["reranker"][reranker_type])

    # ── Retrieval extras (only adapters used by the selected retriever) ────
    retrieval_type = config.retrieval.type
    if retrieval_type in matrix.get("retrieval", {}):
        deps.update(matrix["retrieval"][retrieval_type])

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_type = config.generation.llm.type
    if llm_type in matrix["llm"]:
        deps.update(matrix["llm"][llm_type])

    # ── Chunking extras ───────────────────────────────────────────────────
    chunking_type = config.indexing.chunking.type
    deps.update(CHUNKING_EXTRA_DEPS.get(chunking_type, []))
    if chunking_type == "semantic" and framework == "langchain":
        deps.add("langchain-experimental>=0.4,<0.5")

    # ── Parser extras ─────────────────────────────────────────────────────
    parser_value = config.ingestion.parser  # str value
    deps.update(PARSER_EXTRA_DEPS.get(parser_value, []))

    # ── Evaluation extras (optional) ──────────────────────────────────────
    if config.evaluation is not None:
        eval_framework = config.evaluation.framework  # str value
        deps.update(EVALUATION_EXTRA_DEPS.get(eval_framework, []))
        if eval_framework == "deepeval":
            deps.add("openai>=1.40.0,<3.0.0")

    return sorted(deps)
