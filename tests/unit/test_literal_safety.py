"""Config values must remain data in generated vector-store source."""

import ast

import pytest

from ragfactory.core.generator import TemplateLoader


@pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
@pytest.mark.parametrize(
    ("store", "fields"),
    [
        ("chromadb", ["persist_directory", "collection_name"]),
        ("qdrant", ["url", "collection_name"]),
        ("pinecone", ["index_name"]),
        ("weaviate", ["url", "class_name"]),
        ("milvus", ["uri", "collection_name"]),
        ("pgvector", ["collection_name"]),
    ],
)
def test_vector_store_values_are_python_string_literals(framework, store, fields):
    payload = '"); raise RuntimeError("injected") #\n'
    rendered = TemplateLoader().render_stage(
        "vectordb",
        store,
        {
            "framework": framework,
            "config": {"retrieval": {"type": "dense"}},
            "vector_db": dict.fromkeys(fields, payload),
            "embedding_dim": 1536,
        },
    )
    tree = ast.parse(rendered)
    constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert payload in constants
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(tree))


@pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
@pytest.mark.parametrize("reranker", ["cohere", "cross_encoder", "colbert", "flashrank"])
def test_reranker_model_values_are_python_string_literals(framework, reranker):
    payload = '"); raise RuntimeError("injected") #\n'
    rendered = TemplateLoader().render_stage(
        "reranker",
        reranker,
        {"framework": framework, "reranker": {"model": payload, "top_n": 5}},
    )
    tree = ast.parse(rendered)
    assert payload in [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(tree))
