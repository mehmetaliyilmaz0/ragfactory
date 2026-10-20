import ast

import pytest

from ragfactory.core.config import (
    GenerationConfig,
    IndexingConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    QdrantConfig,
    RAGPipelineConfig,
    SemanticChunkingConfig,
)
from ragfactory.core.generator import generate


@pytest.mark.parametrize("threshold", ["percentile", "standard_deviation"])
def test_explicit_semantic_threshold_renders_and_uses_selected_embedder(threshold):
    config = RAGPipelineConfig(
        name="semantic-test",
        indexing=IndexingConfig(
            chunking=SemanticChunkingConfig(breakpoint_threshold_type=threshold),
            embedding=OpenAIEmbeddingConfig(),
            vector_db=QdrantConfig(),
        ),
        generation=GenerationConfig(llm=OpenAILLMConfig()),
    )
    result = generate(config)
    assert result.validation_passed, result.errors
    source = result.files["ingestion.py"]
    tree = ast.parse(source)
    chunker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_chunking"
    )
    call = next(
        node
        for node in ast.walk(chunker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SemanticChunker"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(keywords["breakpoint_threshold_type"]) == threshold
    assert isinstance(keywords["embeddings"], ast.Call)
    assert keywords["embeddings"].func.id == "build_embedding"
    assert ast.literal_eval(keywords["min_chunk_size"]) == 100
