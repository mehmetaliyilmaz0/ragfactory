from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ragfactory.core.config import (
    EvaluationConfig,
    GenerationConfig,
    HyDEConfig,
    IndexingConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    QdrantConfig,
    QueryRewritingConfig,
    RAGPipelineConfig,
)
from ragfactory.core.generator import TemplateLoader, _collect_required_env_vars
from ragfactory.core.versions import get_dependencies


def _config(framework: str = "langchain") -> RAGPipelineConfig:
    return RAGPipelineConfig(
        name="runtime-refresh",
        framework=framework,
        indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
        generation=GenerationConfig(llm=OpenAILLMConfig()),
    )


def _ctx(config: RAGPipelineConfig) -> dict[str, Any]:
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "env_vars": _collect_required_env_vars(config),
    }


def _loader() -> TemplateLoader:
    from ragfactory.core.generator import _DEFAULT_TEMPLATE_DIR

    return TemplateLoader(_DEFAULT_TEMPLATE_DIR)


def test_generated_chunking_separator_is_emitted_as_python_literal() -> None:
    config = _config()
    config.indexing.chunking.separators = ['"\nraise RuntimeError("injected")']
    rendered = _loader().render_stage(
        "chunking", "recursive", {**_ctx(config), "chunking": config.indexing.chunking}
    )
    assert 'separators=[\'"\\nraise RuntimeError("injected")\']' in rendered


def test_generated_local_runtime_loads_dotenv() -> None:
    config = _config()
    rendered = _loader().render_entrypoint("langchain", "pipeline", _pipeline_context(config))
    assert "from dotenv import load_dotenv" in rendered
    assert "load_dotenv()" in rendered


def test_generated_api_offloads_sync_pipeline_calls() -> None:
    rendered = _loader().render_common("api.py", _ctx(_config()))
    assert "async def query_rag" in rendered
    assert "res = await asyncio.to_thread(" in rendered
    assert "run_pipeline, request.query" in rendered


def test_docker_build_copies_readme_and_modules_before_install() -> None:
    rendered = _loader().render_common("Dockerfile", _ctx(_config()))
    assert rendered.index("COPY pyproject.toml README.md") < rendered.index("RUN pip install")
    assert rendered.index("COPY *.py") < rendered.index("RUN pip install")


def test_generated_pyproject_selects_flat_runtime_modules() -> None:
    rendered = _loader().render_common("pyproject.toml", _ctx(_config()))
    assert 'only-include = ["pipeline.py", "ingestion.py", "api.py", "eval.py"]' in rendered


def test_llamaindex_pipeline_applies_the_configured_prompt() -> None:
    config = _config("llamaindex")
    rendered = _loader().render_entrypoint("llamaindex", "pipeline", _pipeline_context(config))
    assert 'template_var_mappings={"context_str": "context", "query_str": "question"}' in rendered
    assert "get_response_synthesizer" in rendered
    assert "async def run_pipeline_async" in rendered


def test_llamaindex_composes_query_rewriting_and_hyde() -> None:
    config = _config("llamaindex")
    config.pre_retrieval.query_rewriting = QueryRewritingConfig()
    config.pre_retrieval.hyde = HyDEConfig()
    rendered = _loader().render_entrypoint("llamaindex", "pipeline", _pipeline_context(config))
    assert "retriever = QueryFusionRetriever(" in rendered
    assert "_hyde_transform = HyDEQueryTransform(include_original=True)" in rendered
    assert "TransformQueryEngine(query_engine, query_transform=_hyde_transform)" in rendered
    assert "elif pre_retrieval.hyde" not in rendered
    function = next(
        node
        for node in ast.parse(rendered).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_pipeline_async"
    )
    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "QueryFusionRetriever" in calls
    assert "HyDEQueryTransform" in calls
    assert "TransformQueryEngine" in calls


def test_llamaindex_agentic_sync_wrapper_uses_run_pipeline_async_safely() -> None:
    rendered = _loader().render_entrypoint(
        "llamaindex", "pipeline_agentic", _pipeline_context(_config("llamaindex"))
    )
    assert "asyncio.run(run_pipeline_async(query, return_contexts))" in rendered
    assert "call run_pipeline_async from an existing event loop" in rendered
    assert "get_event_loop" not in rendered


def test_agentic_entrypoints_apply_rerankers_and_keep_async_retrieval() -> None:
    config = _config("llamaindex")
    context = _pipeline_context(config)
    context["stages"]["reranker"] = "def build_reranker(): return object()"
    llama = _loader().render_entrypoint("llamaindex", "pipeline_agentic", context)
    assert "await retriever.aretrieve(ev.query)" in llama
    assert "await build_reranker().apostprocess_nodes(nodes, query_str=ev.query)" in llama

    config = _config("langchain")
    context = _pipeline_context(config)
    context["stages"]["reranker"] = "def build_reranker(retriever): return retriever"
    langchain = _loader().render_entrypoint("langchain", "pipeline_agentic", context)
    assert "retriever = build_reranker(retriever)" in langchain
    assert 'block.get("text", "")' in langchain
    assert 'block.get("type") == "text"' in langchain


def test_langchain_directory_ingestion_uses_stdlib_text_loading() -> None:
    config = _config()
    rendered = _loader().render_entrypoint(
        "langchain",
        "ingestion",
        {
            **_ctx(config),
            "ingestion": config.ingestion,
            "stages": {"chunking": "", "embedding": "", "vectordb": ""},
        },
    )
    assert "langchain_core.documents import Document" in rendered
    assert "path.glob(glob)" in rendered


def test_langchain_ingestion_persists_final_chunk_corpus_for_hybrid_search() -> None:
    config = _config()
    rendered = _loader().render_entrypoint(
        "langchain",
        "ingestion",
        {
            **_ctx(config),
            "ingestion": config.ingestion,
            "stages": {"chunking": "", "embedding": "", "vectordb": ""},
        },
    )
    assert 'Path(__file__).with_name("chunks.json")' in rendered
    assert '"page_content": chunk.page_content' in rendered
    assert '"metadata": chunk.metadata' in rendered
    assert "corpus.extend(" in rendered


def test_langchain_ingestion_sidecar_accumulates_multiple_sources(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    documents = ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content: str, metadata: dict[str, str]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    documents.Document = Document
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents)
    stages = """
class _Chunker:
    def split_documents(self, docs):
        return docs
def build_chunking():
    return _Chunker()
def build_embedding():
    return None
class _Store:
    def add_documents(self, docs):
        pass
def build_vectordb(embedder):
    return _Store()
"""
    config = _config()
    rendered = _loader().render_entrypoint(
        "langchain",
        "ingestion",
        {
            **_ctx(config),
            "ingestion": config.ingestion,
            "stages": {"chunking": stages, "embedding": "", "vectordb": ""},
        },
    )
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "first.txt").write_text("first chunk", encoding="utf-8")
    script = tmp_path / "ingestion.py"
    module = ModuleType("generated_ingestion")
    module.__dict__["__file__"] = str(script)
    exec(compile(rendered, str(script), "exec"), module.__dict__)
    module.ingest(str(source_dir))
    (source_dir / "second.txt").write_text("second chunk", encoding="utf-8")
    module.ingest(str(source_dir / "second.txt"))
    corpus = __import__("json").loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert [item["page_content"] for item in corpus] == ["first chunk", "second chunk"]


def _eval_context(framework: str = "ragas") -> dict[str, Any]:
    config = _config()
    config.evaluation = EvaluationConfig(framework=framework, judge_model="judge-model-from-config")
    return {**_ctx(config), "evaluation": config.evaluation}


def test_ragas_eval_uses_current_metric_api_and_configured_judge() -> None:
    rendered = _loader().render_common("eval.py", _eval_context())
    assert "from ragas.metrics.collections import" in rendered
    assert "llm_factory(_JUDGE_MODEL" in rendered
    assert "from ragas.embeddings.base import embedding_factory" in rendered
    assert "AnswerRelevancy(llm=judge, embeddings=relevancy_embeddings)" in rendered
    assert ".ascore(" in rendered
    assert "math.isfinite(value)" in rendered


def test_deepeval_eval_passes_configured_model_to_each_metric() -> None:
    rendered = _loader().render_common("eval.py", _eval_context("deepeval"))
    assert "model=_JUDGE_MODEL" in rendered
    assert "from deepeval.metrics import" in rendered


def test_eval_rejects_malformed_dataset_instead_of_using_defaults() -> None:
    rendered = _loader().render_common("eval.py", _eval_context())
    assert "except (OSError, json.JSONDecodeError) as exc" in rendered
    assert "raise ValueError" in rendered
    assert "Using defaults" not in rendered


def _load_eval_module(monkeypatch: Any, framework: str = "ragas") -> ModuleType:
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    pipeline = ModuleType("pipeline")
    pipeline.run_pipeline = lambda *_args, **_kwargs: ("answer", ["context"])
    monkeypatch.setitem(sys.modules, "pipeline", pipeline)
    module = ModuleType("generated_eval")
    module.__dict__["__name__"] = "generated_eval"
    exec(
        compile(_loader().render_common("eval.py", _eval_context(framework)), "eval.py", "exec"),
        module.__dict__,
    )
    return module


def test_eval_score_gate_rejects_nan_infinity_and_low_scores(monkeypatch: Any) -> None:
    module = _load_eval_module(monkeypatch)
    assert module.score_passes("faithfulness", 0.9)
    assert not module.score_passes("faithfulness", float("nan"))
    assert not module.score_passes("faithfulness", float("inf"))
    assert not module.score_passes("faithfulness", 0.1)


def test_eval_dataset_malformed_json_raises_explicitly(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_eval_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    Path("eval_dataset.json").write_text("{invalid", encoding="utf-8")
    try:
        module.load_test_cases()
    except ValueError as exc:
        assert "Cannot read eval_dataset.json" in str(exc)
    else:
        raise AssertionError("malformed evaluation dataset should fail explicitly")


def test_ragas_evaluator_uses_configured_judge_and_inputs(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}
    openai = ModuleType("openai")
    openai.AsyncOpenAI = lambda: object()
    ragas_embeddings = ModuleType("ragas.embeddings.base")
    ragas_embeddings.embedding_factory = lambda provider, **kwargs: (
        calls.update(embedding_provider=provider, embedding_kwargs=kwargs) or "embeddings"
    )
    ragas_llms = ModuleType("ragas.llms")
    ragas_llms.llm_factory = lambda model, client: (
        calls.update(model=model, client=client) or "judge"
    )
    collections = ModuleType("ragas.metrics.collections")

    class Metric:
        def __init__(self, **kwargs: Any) -> None:
            calls["metric_kwargs"] = kwargs

        async def ascore(self, **kwargs: Any) -> Any:
            calls["sample"] = kwargs
            return type("Score", (), {"value": 0.9})()

    collections.AnswerRelevancy = Metric
    collections.ContextPrecision = Metric
    collections.Faithfulness = Metric
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", ragas_embeddings)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)
    module = _load_eval_module(monkeypatch)
    assert module._run_ragas(
        [
            {
                "question": "question",
                "answer": "answer",
                "contexts": ["context"],
                "ground_truth": "reference",
            }
        ]
    )
    assert calls["model"] == "judge-model-from-config"
    assert calls["embedding_provider"] == "openai"
    assert calls["embedding_kwargs"]["model"] == "text-embedding-3-small"
    assert calls["embedding_kwargs"]["client"] is not None


def test_ragas_metrics_receive_only_supported_inputs(monkeypatch: Any) -> None:
    received: list[dict[str, Any]] = []
    openai = ModuleType("openai")
    openai.AsyncOpenAI = lambda: object()
    embeddings = ModuleType("ragas.embeddings.base")
    embeddings.embedding_factory = lambda *_args, **_kwargs: "embeddings"
    ragas_llms = ModuleType("ragas.llms")
    ragas_llms.llm_factory = lambda *_args, **_kwargs: "judge"
    collections = ModuleType("ragas.metrics.collections")

    class Metric:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def ascore(self, **kwargs: Any) -> Any:
            received.append(kwargs)
            return type("Score", (), {"value": 0.9})()

    collections.AnswerRelevancy = Metric
    collections.ContextPrecision = Metric
    collections.Faithfulness = Metric
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", embeddings)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)
    module = _load_eval_module(monkeypatch)
    assert module._run_ragas(
        [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}]
    )
    assert received == [
        {"user_input": "q", "response": "a", "retrieved_contexts": ["c"]},
        {"user_input": "q", "response": "a"},
        {"user_input": "q", "response": "a", "reference": "g", "retrieved_contexts": ["c"]},
    ]


def test_deepeval_evaluator_passes_configured_model_to_metrics(monkeypatch: Any) -> None:
    models: list[str] = []
    deepeval_metrics = ModuleType("deepeval.metrics")
    deepeval_cases = ModuleType("deepeval.test_case")
    deepeval_cases.LLMTestCase = lambda **kwargs: kwargs

    class Metric:
        def __init__(self, threshold: float, model: str) -> None:
            models.append(model)
            self.score = 0.9

        def measure(self, case: dict[str, Any]) -> None:
            return None

        def is_successful(self) -> bool:
            return True

    deepeval_metrics.AnswerRelevancyMetric = Metric
    deepeval_metrics.ContextualPrecisionMetric = Metric
    deepeval_metrics.FaithfulnessMetric = Metric
    monkeypatch.setitem(sys.modules, "deepeval.metrics", deepeval_metrics)
    monkeypatch.setitem(sys.modules, "deepeval.test_case", deepeval_cases)
    module = _load_eval_module(monkeypatch, "deepeval")
    assert module._run_deepeval(
        [
            {
                "question": "question",
                "answer": "answer",
                "contexts": ["context"],
                "ground_truth": "reference",
            }
        ]
    )
    assert models == ["judge-model-from-config"] * 3


def _pipeline_context(config: RAGPipelineConfig) -> dict[str, Any]:
    return {
        **_ctx(config),
        "generation": config.generation,
        "chunking": config.indexing.chunking,
        "embedding": config.indexing.embedding,
        "vector_db": config.indexing.vector_db,
        "retrieval": config.retrieval,
        "pre_retrieval": config.pre_retrieval,
        "post_retrieval": config.post_retrieval,
        "stages": {
            "chunking": "",
            "embedding": "",
            "vectordb": "",
            "retrieval": "",
            "llm": "",
            "reranker": "",
        },
    }
