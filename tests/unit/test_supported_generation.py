"""Schema-only features must never produce a silently incomplete project."""

from __future__ import annotations

import pytest

from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.generator import generate
from ragfactory.core.validator import validate


@pytest.mark.parametrize("chunking", ["contextual", "proposition", "late"])
def test_unimplemented_chunking_is_rejected_before_files_are_emitted(chunking: str):
    config = RAGPipelineConfig.model_validate(
        {
            "name": "unsupported-stage",
            "indexing": {
                "chunking": {"type": chunking},
                "embedding": {"type": "jina", "model": "jina-embeddings-v3", "late_chunking": True},
                "vector_db": {"type": "qdrant"},
            },
            "generation": {"llm": {"type": "openai"}},
        }
    )
    validation = validate(config)
    assert not validation.valid
    assert any(issue.code == "UNSUPPORTED_CHUNKING" for issue in validation.errors)
    result = generate(config)
    assert not result.validation_passed
    assert result.files == {}
    assert any("indexing.chunking" in error for error in result.errors)


@pytest.mark.parametrize("store", ["weaviate", "milvus", "pgvector"])
def test_llamaindex_hybrid_requires_implemented_sparse_store(store):
    config = RAGPipelineConfig.model_validate(
        {
            "name": "unsupported-hybrid",
            "framework": "llamaindex",
            "indexing": {"embedding": {"type": "openai"}, "vector_db": {"type": store}},
            "retrieval": {"type": "hybrid_rrf"},
            "generation": {"llm": {"type": "openai"}},
        }
    )
    assert not validate(config).valid
    result = generate(config)
    assert not result.validation_passed
    assert not result.files
