from __future__ import annotations

from ragfactory.core.config import (
    BGEM3EmbeddingConfig,
    CrossEncoderRerankerConfig,
    EvaluationConfig,
    GenerationConfig,
    HybridRRFConfig,
    IndexingConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    PostRetrievalConfig,
    QdrantConfig,
    RAGPipelineConfig,
)
from ragfactory.core.versions import get_dependencies


def _config(**overrides: object) -> RAGPipelineConfig:
    values: dict[str, object] = {
        "name": "refresh-test",
        "indexing": IndexingConfig(
            embedding=OpenAIEmbeddingConfig(),
            vector_db=QdrantConfig(),
        ),
        "generation": GenerationConfig(llm=OpenAILLMConfig()),
    }
    values.update(overrides)
    return RAGPipelineConfig(**values)  # type: ignore[arg-type]


def test_langchain_dependencies_use_v1_classic_and_current_adapters() -> None:
    deps = get_dependencies(_config())

    assert "langchain>=1.4,<2" in deps
    assert "langchain-core>=1,<2" in deps
    assert "langchain-classic>=1,<2" in deps
    assert "langgraph>=1,<2" not in deps
    assert "langchain-openai>=1,<2" in deps
    assert "langchain-qdrant>=1,<2" in deps
    assert "pinecone-client" not in deps
    assert "python-dotenv>=1,<2" in deps


def test_agentic_langchain_adds_langgraph_v1() -> None:
    deps = get_dependencies(_config(flow_type="agentic"))

    assert "langgraph>=1,<2" in deps


def test_langchain_hybrid_retrieval_adds_bm25_runtime() -> None:
    deps = get_dependencies(_config(retrieval=HybridRRFConfig()))

    assert "rank-bm25>=0.2,<0.3" in deps


def test_llamaindex_qdrant_hybrid_enables_fastembed_extra() -> None:
    deps = get_dependencies(_config(framework="llamaindex", retrieval=HybridRRFConfig()))

    assert "llama-index-vector-stores-qdrant[fastembed]>=0.10,<0.11" in deps


def test_llamaindex_uses_core_and_selected_modern_integrations() -> None:
    deps = get_dependencies(_config(framework="llamaindex"))

    assert "llama-index-core>=0.14,<0.15" in deps
    assert "llama-index-llms-openai>=0.8,<0.9" in deps
    assert "llama-index-embeddings-openai>=0.7,<0.8" in deps
    assert "llama-index-vector-stores-qdrant>=0.10,<0.11" in deps
    assert not any(dep.startswith("llama-index>=") for dep in deps)


def test_eval_dependencies_use_current_compatible_major_lines() -> None:
    deps = get_dependencies(_config(evaluation=EvaluationConfig(framework="both")))

    assert "ragas>=0.4,<0.5" in deps
    assert "langchain-community>=0.4,<0.4.2" in deps
    assert "deepeval>=4,<5" in deps
    assert "openai>=1.40.0,<3.0.0" in deps


def test_local_huggingface_adapters_use_package_compatible_dependencies() -> None:
    langchain = get_dependencies(
        _config(
            indexing=IndexingConfig(embedding=BGEM3EmbeddingConfig(), vector_db=QdrantConfig()),
            post_retrieval=PostRetrievalConfig(reranker=CrossEncoderRerankerConfig()),
        )
    )
    llamaindex = get_dependencies(
        _config(
            framework="llamaindex",
            indexing=IndexingConfig(embedding=BGEM3EmbeddingConfig(), vector_db=QdrantConfig()),
        )
    )

    assert "langchain-huggingface>=1.2,<2" in langchain
    assert "langchain-community>=0.4,<0.5" in langchain
    assert "FlagEmbedding>=1.2.0,<2.0.0" not in langchain
    assert "llama-index-embeddings-huggingface>=0.6,<1" in llamaindex
    assert not any(dep.startswith("FlagEmbedding") for dep in llamaindex)
