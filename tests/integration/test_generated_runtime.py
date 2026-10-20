"""Run generated projects against real SDKs and a local Qdrant database.

CI installs each generated project's dependencies before enabling these tests.
No model downloads or paid API requests are needed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.generator import generate


@pytest.mark.skipif(
    not os.getenv("RAGFACTORY_SDK_SMOKE"), reason="Run in the generated project's SDK environment"
)
@pytest.mark.parametrize("retrieval", ["dense", "hybrid_rrf", "hybrid_weighted"])
def test_generated_pipeline_with_real_sdks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retrieval
):
    framework = os.environ["RAGFACTORY_SDK_SMOKE"]
    retrieval_config = {"type": retrieval, "top_k": 1}
    if framework == "langchain" and retrieval != "dense":
        retrieval_config.update(bm25_k1=2.2, bm25_b=0.3)
    config = RAGPipelineConfig.model_validate(
        {
            "name": "sdk-smoke",
            "framework": framework,
            "indexing": {"embedding": {"type": "openai"}, "vector_db": {"type": "qdrant"}},
            "retrieval": retrieval_config,
            "generation": {
                "llm": {"type": "openai"},
                "prompt_template": "CUSTOM PROMPT\n{context}\nQuestion: {question}",
            },
        }
    )
    result = generate(config)
    assert result.validation_passed, result.errors
    modules = {}
    for name in ("pipeline", "ingestion"):
        source = tmp_path / f"{name}.py"
        source.write_text(result.files[f"{name}.py"], encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"generated_smoke_{name}", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    pipeline, ingestion = modules["pipeline"], modules["ingestion"]
    sparse_calls = []

    # Constructors catch unsupported model IDs and obsolete provider arguments.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-network")
    pipeline.build_embedding()
    configured_llm = pipeline.build_llm()
    if framework == "llamaindex":
        assert configured_llm.metadata.context_window > 0

    if framework == "langchain":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient, models

        embedder = DeterministicFakeEmbedding(size=8)
        client = QdrantClient(":memory:")
        client.create_collection(
            "smoke", vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE)
        )
        store = QdrantVectorStore(client=client, collection_name="smoke", embedding=embedder)
        llm = FakeListChatModel(responses=["Standalone RAG projects."])
    else:
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.llms import MockLLM
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import AsyncQdrantClient, QdrantClient

        embedder = MockEmbedding(embed_dim=8)
        database_path = str(tmp_path / "qdrant")
        sparse_options = {}
        if retrieval != "dense":
            # Replace only sparse model inference; Qdrant indexing/query/fusion stay real.
            def encode_sparse(texts):
                sparse_calls.append(texts)
                return [[0] for _ in texts], [[1.0] for _ in texts]

            sparse_options = {
                "enable_hybrid": True,
                "fastembed_sparse_model": "Qdrant/bm25",
                "sparse_doc_fn": encode_sparse,
                "sparse_query_fn": encode_sparse,
            }
            if retrieval == "hybrid_rrf":
                sparse_options["hybrid_fusion_fn"] = pipeline._rrf_fusion
        store = QdrantVectorStore(
            client=QdrantClient(path=database_path), collection_name="smoke", **sparse_options
        )
        llm = MockLLM()

    monkeypatch.setattr(ingestion, "build_embedding", lambda: embedder)
    monkeypatch.setattr(ingestion, "build_vectordb", lambda _: store)
    document = tmp_path / "document.txt"
    document.write_text("Ragfactory generates standalone RAG projects.", encoding="utf-8")
    ingestion.ingest(str(document))

    if framework == "llamaindex":
        store.client.close()
        store = QdrantVectorStore(
            aclient=AsyncQdrantClient(path=database_path), collection_name="smoke", **sparse_options
        )

    monkeypatch.setattr(pipeline, "build_embedding", lambda: embedder)
    monkeypatch.setattr(pipeline, "build_vectordb", lambda _: store)
    monkeypatch.setattr(pipeline, "build_llm", lambda: llm)
    if framework == "langchain" and retrieval != "dense":
        bm25 = pipeline.build_retrieval(store).retrievers[0].vectorizer
        assert bm25.k1 == 2.2
        assert bm25.b == 0.3
    answer, contexts = pipeline.run_pipeline("What does Ragfactory generate?", return_contexts=True)
    assert answer
    assert contexts == ["Ragfactory generates standalone RAG projects."]
    if framework == "llamaindex":
        assert "CUSTOM PROMPT" in answer
        if retrieval != "dense":
            assert len(sparse_calls) >= 2, "Hybrid must encode both documents and the query"
