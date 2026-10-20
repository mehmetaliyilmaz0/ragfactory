from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from ragfactory.core.config import (
    AnthropicLLMConfig,
    GenerationConfig,
    HybridRRFConfig,
    IndexingConfig,
    OpenAIEmbeddingConfig,
    QdrantConfig,
    RAGPipelineConfig,
)
from ragfactory.core.generator import TemplateLoader


@dataclass
class _Node:
    node_id: str


@dataclass
class _Result:
    nodes: list[_Node] | None
    similarities: list[float] | None = None
    ids: list[str] | None = None


def _rrf_function():
    config = RAGPipelineConfig(
        name="rrf-fusion-test",
        framework="llamaindex",
        indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
        retrieval=HybridRRFConfig(top_k=3, rrf_k=10),
        generation=GenerationConfig(llm=AnthropicLLMConfig()),
    )
    rendered = TemplateLoader().render_stage(
        "vectordb",
        "qdrant",
        {
            "config": config,
            "framework": "llamaindex",
            "vector_db": config.indexing.vector_db,
            "vector_db_type": "qdrant",
        },
    )
    tree = ast.parse(rendered)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rrf_fusion"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"Any": object, "VectorStoreQueryResult": _Result}
    exec(compile(module, "generated_qdrant.py", "exec"), namespace)
    return namespace["_rrf_fusion"]


def test_llamaindex_qdrant_rrf_fuses_duplicates_and_limits_results() -> None:
    fuse = _rrf_function()

    result = fuse(
        _Result([_Node("shared"), _Node("dense-only"), _Node("other")]),
        _Result([_Node("shared"), _Node("sparse-only")]),
        alpha=0.5,
        top_k=3,
    )

    assert result.ids == ["shared", "dense-only", "sparse-only"]
    assert result.nodes[0].node_id == "shared"
    assert result.similarities == pytest.approx([2 / 11, 1 / 12, 1 / 12])


def test_llamaindex_qdrant_rrf_accepts_missing_retriever_results() -> None:
    fuse = _rrf_function()

    result = fuse(_Result(None), _Result([_Node("sparse-only")]), alpha=0.5, top_k=2)

    assert result.ids == ["sparse-only"]
