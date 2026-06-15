"""
Unit tests for ragfactory Jinja2 templates (Phase 1c).

Unlike test_generator.py (which uses stub templates to test the engine),
this file uses the REAL templates from ragfactory/templates/ to verify
that generated code is correct, complete, and production-grade.

Test classes mirror implementation phases:
  Phase 0:  TestCommonTemplates, TestDockerCompose
  Phase 1:  TestLangChainEntrypoints
  Phase 2:  TestLlamaIndexEntrypoints
  Phase 3:  TestChunkingStages
  Phase 4:  TestEmbeddingStages
  Phase 5:  TestVectorDBStages
  Phase 6:  TestRetrievalStages
  Phase 7:  TestRerankerStages
  Phase 8:  TestLLMStages
  Phase 9:  TestEndToEnd, TestEnvVarGeneration, TestDependencyAccuracy
  Phase 10: TestImportReachability, TestConfigYamlRoundTrip,
            TestStrictUndefined, TestASTValidation, TestDockerComposeAbsence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragfactory.core.config import (
    AdvancedGenerationConfig,
    AnthropicLLMConfig,
    BGEM3EmbeddingConfig,
    ChromaDBConfig,
    CohereEmbeddingConfig,
    CohereLLMConfig,
    CohereRerankerConfig,
    ColBERTRerankerConfig,
    ContextualChunkingConfig,
    CRAGConfig,
    CrossEncoderRerankerConfig,
    DenseRetrievalConfig,
    EvaluationConfig,
    FixedChunkingConfig,
    FlashRankRerankerConfig,
    GeminiEmbeddingConfig,
    GenerationConfig,
    HybridRRFConfig,
    HybridWeightedConfig,
    HyDEConfig,
    IndexingConfig,
    JinaEmbeddingConfig,
    LateChunkingConfig,
    MilvusConfig,
    NomicEmbeddingConfig,
    OllamaLLMConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    PageLevelChunkingConfig,
    PgVectorConfig,
    PineconeConfig,
    PostRetrievalConfig,
    PreRetrievalConfig,
    PropositionChunkingConfig,
    QdrantConfig,
    QueryRewritingConfig,
    RAGPipelineConfig,
    RecursiveChunkingConfig,
    SemanticChunkingConfig,
    SentenceWindowConfig,
    SmallToBigConfig,
    VoyageEmbeddingConfig,
    WeaviateConfig,
)
from ragfactory.core.generator import (
    _DEFAULT_TEMPLATE_DIR,
    _EXTERNAL_SERVICE_DBS,
    GeneratorResult,
    TemplateLoader,
    _collect_required_env_vars,
    _get_embedding_dim,
    generate,
)
from ragfactory.core.versions import get_dependencies

# ─── Constants ────────────────────────────────────────────────────────────────

REAL_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "ragfactory" / "templates"


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _make_config(**kwargs: Any) -> RAGPipelineConfig:
    """
    Build a minimal valid RAGPipelineConfig.

    Defaults: langchain, OpenAI embedding, Qdrant vector DB, OpenAI LLM.
    Any field can be overridden via kwargs.
    """
    defaults: dict[str, Any] = {
        "name": "test-pipeline",
        "framework": "langchain",
        "indexing": IndexingConfig(
            embedding=OpenAIEmbeddingConfig(),
            vector_db=QdrantConfig(),
        ),
        "generation": GenerationConfig(llm=OpenAILLMConfig()),
    }
    defaults.update(kwargs)
    return RAGPipelineConfig(**defaults)  # type: ignore[arg-type]


def _loader() -> TemplateLoader:
    """Return a TemplateLoader pointed at the real template directory."""
    return TemplateLoader(REAL_TEMPLATE_DIR)


def _common_ctx(config: RAGPipelineConfig) -> dict[str, Any]:
    """
    Build the context dict for common templates
    (pyproject.toml, .env.example, README.md, Dockerfile).
    """
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "env_vars": _collect_required_env_vars(config),
    }


def _docker_ctx(config: RAGPipelineConfig) -> dict[str, Any]:
    """
    Build the context dict for docker-compose.yml.j2.
    Uses base_ctx + vector_db (no env_vars).
    """
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "vector_db": config.indexing.vector_db,
    }


# ─── Phase 0: Common Templates ────────────────────────────────────────────────


class TestCommonTemplates:
    """
    Tests for entrypoints/common/:
      pyproject.toml.j2, .env.example.j2, README.md.j2, Dockerfile.j2
    """

    # ── pyproject.toml ───────────────────────────────────────────────────────

    def test_pyproject_renders(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("pyproject.toml", _common_ctx(config))
        assert rendered.strip(), "pyproject.toml rendered empty"

    def test_pyproject_has_pipeline_name(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("pyproject.toml", _common_ctx(config))
        assert 'name = "test-pipeline"' in rendered

    def test_pyproject_has_all_dependencies(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("pyproject.toml", _common_ctx(config))
        for dep in get_dependencies(config):
            assert dep in rendered, f"Expected dep {dep!r} missing from pyproject.toml"

    def test_pyproject_has_build_system(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("pyproject.toml", _common_ctx(config))
        assert "[build-system]" in rendered

    def test_pyproject_has_python_version(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("pyproject.toml", _common_ctx(config))
        assert "3.11" in rendered

    # ── .env.example ─────────────────────────────────────────────────────────

    def test_env_example_renders(self) -> None:
        config = _make_config()
        rendered = _loader().render_common(".env.example", _common_ctx(config))
        assert rendered.strip(), ".env.example rendered empty"

    def test_env_example_has_openai_key(self) -> None:
        # Default config uses OpenAI embedding → OPENAI_API_KEY expected
        config = _make_config()
        rendered = _loader().render_common(".env.example", _common_ctx(config))
        assert "OPENAI_API_KEY" in rendered

    def test_env_example_has_qdrant_key(self) -> None:
        # Default config uses Qdrant → QDRANT_API_KEY expected
        config = _make_config()
        rendered = _loader().render_common(".env.example", _common_ctx(config))
        assert "QDRANT_API_KEY" in rendered

    def test_env_example_no_entries_for_local_only_config(self) -> None:
        # BGE-M3 (self-hosted) + ChromaDB (local) + Ollama (local) → no API keys
        config = _make_config(
            indexing=IndexingConfig(
                embedding=BGEM3EmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
            generation=GenerationConfig(llm=OllamaLLMConfig()),
        )
        rendered = _loader().render_common(".env.example", _common_ctx(config))
        # No KEY= lines should be present
        assert "OPENAI_API_KEY" not in rendered
        assert "QDRANT_API_KEY" not in rendered
        assert "ANTHROPIC_API_KEY" not in rendered

    # ── README.md ─────────────────────────────────────────────────────────────

    def test_readme_renders(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("README.md", _common_ctx(config))
        assert rendered.strip(), "README.md rendered empty"

    def test_readme_has_pipeline_name(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("README.md", _common_ctx(config))
        assert "test-pipeline" in rendered

    # ── Dockerfile ────────────────────────────────────────────────────────────

    def test_dockerfile_renders(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("Dockerfile", _common_ctx(config))
        assert rendered.strip(), "Dockerfile rendered empty"

    def test_dockerfile_has_python_version(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("Dockerfile", _common_ctx(config))
        assert "python:3.11" in rendered

    def test_dockerfile_runs_pip_install(self) -> None:
        config = _make_config()
        rendered = _loader().render_common("Dockerfile", _common_ctx(config))
        assert "pip install" in rendered


# ─── Phase 0: docker-compose.yml ──────────────────────────────────────────────


class TestDockerCompose:
    """
    Tests for entrypoints/common/docker-compose.yml.j2.

    docker-compose.yml is only generated for external-service vector DBs:
    qdrant, weaviate, milvus, pgvector.
    chromadb (local) and pinecone (managed SaaS, no docker needed) are excluded.
    """

    def test_docker_compose_renders_for_qdrant(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        rendered = _loader().render_common("docker-compose.yml", _docker_ctx(config))
        assert rendered.strip(), "docker-compose.yml rendered empty for qdrant"

    def test_docker_compose_renders_for_weaviate(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=WeaviateConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        rendered = _loader().render_common("docker-compose.yml", _docker_ctx(config))
        assert rendered.strip()

    def test_docker_compose_renders_for_milvus(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=MilvusConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        rendered = _loader().render_common("docker-compose.yml", _docker_ctx(config))
        assert rendered.strip()

    def test_docker_compose_renders_for_pgvector(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=PgVectorConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        rendered = _loader().render_common("docker-compose.yml", _docker_ctx(config))
        assert rendered.strip()

    def test_docker_compose_qdrant_service_name(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        rendered = _loader().render_common("docker-compose.yml", _docker_ctx(config))
        assert "qdrant:" in rendered

    def test_docker_compose_weaviate_port(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=WeaviateConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        rendered = _loader().render_common("docker-compose.yml", _docker_ctx(config))
        assert "8080" in rendered

    def test_docker_compose_absent_for_chromadb(self) -> None:
        """ChromaDB is an embedded local store — no docker service needed."""
        assert "chromadb" not in _EXTERNAL_SERVICE_DBS

    def test_docker_compose_absent_for_pinecone(self) -> None:
        """Pinecone is a managed SaaS — no docker service needed."""
        assert "pinecone" not in _EXTERNAL_SERVICE_DBS


# ─── Phase 1: LangChain Entrypoints ──────────────────────────────────────────


def _stage_stubs() -> dict[str, str]:
    """Minimal stage fragments for entrypoint testing (phases 1–2).

    Each stub follows the structural contract: module-level import + factory fn.
    Stage templates are tested in depth in phases 3–8; here we only need valid
    Python fragments so the entrypoint template renders syntactically correct code.
    """
    return {
        "chunking": "def build_chunking() -> Any:\n    return None",
        "embedding": "def build_embedding() -> Any:\n    return None",
        "vectordb": "def build_vectordb(embedder: Any) -> Any:\n    return None",
        "retrieval": "def build_retrieval(vectorstore: Any) -> Any:\n    return None",
        "llm": "def build_llm() -> Any:\n    return None",
    }


def _entrypoint_ctx(
    config: RAGPipelineConfig, stages: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build context for pipeline.py.j2 / ingestion.py.j2."""
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "stages": stages if stages is not None else _stage_stubs(),
        "pre_retrieval": config.pre_retrieval,
        "post_retrieval": config.post_retrieval,
        "generation": config.generation,
        "ingestion": config.ingestion,
    }


class TestLangChainEntrypoints:
    """Tests for entrypoints/langchain/pipeline.py.j2 and ingestion.py.j2."""

    # ── pipeline.py ───────────────────────────────────────────────────────────

    def test_langchain_pipeline_renders(self) -> None:
        config = _make_config(framework="langchain")
        rendered = _loader().render_entrypoint("langchain", "pipeline", _entrypoint_ctx(config))
        assert rendered.strip(), "pipeline.py rendered empty"

    def test_langchain_pipeline_has_run_pipeline_fn(self) -> None:
        config = _make_config(framework="langchain")
        rendered = _loader().render_entrypoint("langchain", "pipeline", _entrypoint_ctx(config))
        assert "def run_pipeline(" in rendered

    def test_langchain_pipeline_embeds_stage_functions(self) -> None:
        config = _make_config(framework="langchain")
        rendered = _loader().render_entrypoint("langchain", "pipeline", _entrypoint_ctx(config))
        for fn in (
            "build_chunking",
            "build_embedding",
            "build_vectordb",
            "build_retrieval",
            "build_llm",
        ):
            assert fn in rendered, f"Stage function {fn!r} missing from pipeline.py"

    def test_langchain_pipeline_no_reranker_by_default(self) -> None:
        """When reranker is absent from stages, build_reranker must not appear."""
        config = _make_config(framework="langchain")
        stubs = _stage_stubs()  # no "reranker" key
        rendered = _loader().render_entrypoint(
            "langchain", "pipeline", _entrypoint_ctx(config, stubs)
        )
        assert "build_reranker" not in rendered

    def test_langchain_pipeline_includes_reranker_when_present(self) -> None:
        """When reranker stage is in context, build_reranker() must be called."""
        config = _make_config(framework="langchain")
        stubs = {
            **_stage_stubs(),
            "reranker": "def build_reranker(retriever: Any) -> Any:\n    return retriever",
        }
        rendered = _loader().render_entrypoint(
            "langchain", "pipeline", _entrypoint_ctx(config, stubs)
        )
        assert "build_reranker" in rendered

    def test_langchain_pipeline_has_prompt_template(self) -> None:
        """The default prompt template must be embedded as _PROMPT_TEMPLATE."""
        config = _make_config(framework="langchain")
        rendered = _loader().render_entrypoint("langchain", "pipeline", _entrypoint_ctx(config))
        assert "_PROMPT_TEMPLATE" in rendered
        assert "{context}" in rendered
        assert "{question}" in rendered

    def test_langchain_pipeline_query_rewriting_block(self) -> None:
        """When query_rewriting.enabled=True, the rewriting code block is present."""
        from ragfactory.core.config import PreRetrievalConfig, QueryRewritingConfig

        config = _make_config(
            framework="langchain",
            pre_retrieval=PreRetrievalConfig(query_rewriting=QueryRewritingConfig(enabled=True)),
        )
        rendered = _loader().render_entrypoint("langchain", "pipeline", _entrypoint_ctx(config))
        assert "Query Rewriting" in rendered

    def test_langchain_pipeline_hyde_block(self) -> None:
        """When hyde.enabled=True, the HyDE code block is present."""
        from ragfactory.core.config import HyDEConfig, PreRetrievalConfig

        config = _make_config(
            framework="langchain",
            pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=True)),
        )
        rendered = _loader().render_entrypoint("langchain", "pipeline", _entrypoint_ctx(config))
        assert "HyDE" in rendered

    # ── ingestion.py ──────────────────────────────────────────────────────────

    def test_langchain_ingestion_renders(self) -> None:
        config = _make_config(framework="langchain")
        rendered = _loader().render_entrypoint("langchain", "ingestion", _entrypoint_ctx(config))
        assert rendered.strip(), "ingestion.py rendered empty"

    def test_langchain_ingestion_has_ingest_fn(self) -> None:
        config = _make_config(framework="langchain")
        rendered = _loader().render_entrypoint("langchain", "ingestion", _entrypoint_ctx(config))
        assert "def ingest(" in rendered


# ─── Phase 2: LlamaIndex Entrypoints ─────────────────────────────────────────


class TestLlamaIndexEntrypoints:
    """Tests for entrypoints/llamaindex/pipeline.py.j2 and ingestion.py.j2."""

    # ── pipeline.py ───────────────────────────────────────────────────────────

    def test_llamaindex_pipeline_renders(self) -> None:
        config = _make_config(framework="llamaindex")
        rendered = _loader().render_entrypoint("llamaindex", "pipeline", _entrypoint_ctx(config))
        assert rendered.strip(), "pipeline.py rendered empty"

    def test_llamaindex_pipeline_has_run_pipeline_fn(self) -> None:
        config = _make_config(framework="llamaindex")
        rendered = _loader().render_entrypoint("llamaindex", "pipeline", _entrypoint_ctx(config))
        assert "def run_pipeline(" in rendered

    def test_llamaindex_pipeline_embeds_stage_functions(self) -> None:
        config = _make_config(framework="llamaindex")
        rendered = _loader().render_entrypoint("llamaindex", "pipeline", _entrypoint_ctx(config))
        for fn in (
            "build_chunking",
            "build_embedding",
            "build_vectordb",
            "build_retrieval",
            "build_llm",
        ):
            assert fn in rendered, f"Stage function {fn!r} missing from pipeline.py"

    def test_llamaindex_pipeline_no_reranker_by_default(self) -> None:
        config = _make_config(framework="llamaindex")
        stubs = _stage_stubs()
        rendered = _loader().render_entrypoint(
            "llamaindex", "pipeline", _entrypoint_ctx(config, stubs)
        )
        assert "build_reranker" not in rendered

    def test_llamaindex_pipeline_includes_reranker_when_present(self) -> None:
        config = _make_config(framework="llamaindex")
        stubs = {**_stage_stubs(), "reranker": "def build_reranker() -> Any:\n    return None"}
        rendered = _loader().render_entrypoint(
            "llamaindex", "pipeline", _entrypoint_ctx(config, stubs)
        )
        assert "build_reranker" in rendered

    def test_llamaindex_pipeline_has_prompt_template(self) -> None:
        config = _make_config(framework="llamaindex")
        rendered = _loader().render_entrypoint("llamaindex", "pipeline", _entrypoint_ctx(config))
        assert "_PROMPT_TEMPLATE" in rendered

    def test_llamaindex_pipeline_query_rewriting_block(self) -> None:
        from ragfactory.core.config import PreRetrievalConfig, QueryRewritingConfig

        config = _make_config(
            framework="llamaindex",
            pre_retrieval=PreRetrievalConfig(query_rewriting=QueryRewritingConfig(enabled=True)),
        )
        rendered = _loader().render_entrypoint("llamaindex", "pipeline", _entrypoint_ctx(config))
        assert "Query Rewriting" in rendered

    def test_llamaindex_pipeline_hyde_block(self) -> None:
        from ragfactory.core.config import HyDEConfig, PreRetrievalConfig

        config = _make_config(
            framework="llamaindex",
            pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=True)),
        )
        rendered = _loader().render_entrypoint("llamaindex", "pipeline", _entrypoint_ctx(config))
        assert "HyDE" in rendered

    # ── ingestion.py ──────────────────────────────────────────────────────────

    def test_llamaindex_ingestion_renders(self) -> None:
        config = _make_config(framework="llamaindex")
        rendered = _loader().render_entrypoint("llamaindex", "ingestion", _entrypoint_ctx(config))
        assert rendered.strip(), "ingestion.py rendered empty"

    def test_llamaindex_ingestion_has_ingest_fn(self) -> None:
        config = _make_config(framework="llamaindex")
        rendered = _loader().render_entrypoint("llamaindex", "ingestion", _entrypoint_ctx(config))
        assert "def ingest(" in rendered


# ─── Phase 3: Chunking Stages ─────────────────────────────────────────────────


def _chunking_ctx(config: RAGPipelineConfig) -> dict[str, Any]:
    """Build context dict for a chunking stage template."""
    fw = str(config.framework)
    return {
        "config": config,
        "framework": fw,
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "chunking": config.indexing.chunking,
    }


class TestChunkingStages:
    """Tests for stages/chunking/*.py.j2 — all 7 chunking strategy templates."""

    # ── fixed ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_fixed_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=FixedChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "fixed", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── recursive ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_recursive_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=RecursiveChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "recursive", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── semantic ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_semantic_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=SemanticChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "semantic", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── contextual ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_contextual_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=ContextualChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "contextual", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── late ──────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_late_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=LateChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "late", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── page_level ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_page_level_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=PageLevelChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "page_level", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── proposition ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_proposition_chunking_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=PropositionChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("chunking", "proposition", _chunking_ctx(config))
        assert rendered.strip()
        assert "def build_chunking()" in rendered

    # ── config value interpolation tests ─────────────────────────────────────

    def test_recursive_separators_in_langchain(self) -> None:
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=RecursiveChunkingConfig(chunk_size=1024, separators=["\n\n", "\n", " "]),
            ),
        )
        rendered = _loader().render_stage("chunking", "recursive", _chunking_ctx(config))
        assert "1024" in rendered
        assert "\\n\\n" in rendered

    def test_contextual_context_model_in_output(self) -> None:
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=ContextualChunkingConfig(context_model="claude-3-haiku-20240307"),
            ),
        )
        rendered = _loader().render_stage("chunking", "contextual", _chunking_ctx(config))
        assert "claude-3-haiku-20240307" in rendered

    def test_proposition_extraction_model_in_output(self) -> None:
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=PropositionChunkingConfig(extraction_model="gpt-4o-mini"),
            ),
        )
        rendered = _loader().render_stage("chunking", "proposition", _chunking_ctx(config))
        assert "gpt-4o-mini" in rendered

    def test_late_chunking_has_late_comment(self) -> None:
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=LateChunkingConfig(chunk_size=256),
            ),
        )
        rendered = _loader().render_stage("chunking", "late", _chunking_ctx(config))
        assert "256" in rendered
        assert "Late Chunking" in rendered


# ─── Phase 4: Embedding Stages ────────────────────────────────────────────────


def _embedding_ctx(config: RAGPipelineConfig) -> dict[str, Any]:
    """Build context dict for an embedding stage template."""
    fw = str(config.framework)
    return {
        "config": config,
        "framework": fw,
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "embedding": config.indexing.embedding,
        "is_late_chunking": config.indexing.chunking.type == "late",
    }


class TestEmbeddingStages:
    """Tests for stages/embedding/*.py.j2 — all 7 embedding provider templates."""

    # ── openai ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_openai_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "openai", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── cohere ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_cohere_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=CohereEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "cohere", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── voyage ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_voyage_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=VoyageEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "voyage", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── gemini ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_gemini_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=GeminiEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "gemini", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── bge_m3 ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_bge_m3_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=BGEM3EmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "bge_m3", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── nomic ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_nomic_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=NomicEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "nomic", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── jina ──────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_jina_embedding_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=JinaEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("embedding", "jina", _embedding_ctx(config))
        assert rendered.strip()
        assert "def build_embedding()" in rendered

    # ── specific interpolation tests ──────────────────────────────────────────

    def test_openai_dimensions_none(self) -> None:
        """When dimensions=None, the rendered code must not include 'dimensions'."""
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(dimensions=None), vector_db=QdrantConfig()
            ),
        )
        rendered = _loader().render_stage("embedding", "openai", _embedding_ctx(config))
        assert "dimensions" not in rendered

    def test_openai_dimensions_set(self) -> None:
        """When dimensions=512, '512' must appear in the rendered code."""
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(dimensions=512), vector_db=QdrantConfig()
            ),
        )
        rendered = _loader().render_stage("embedding", "openai", _embedding_ctx(config))
        assert "512" in rendered

    def test_jina_late_chunking_active(self) -> None:
        """When is_late_chunking=True, 'retrieval.passage' must appear in the rendered code."""
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=JinaEmbeddingConfig(late_chunking=True),
                vector_db=QdrantConfig(),
                chunking=LateChunkingConfig(),
            ),
        )
        rendered = _loader().render_stage("embedding", "jina", _embedding_ctx(config))
        assert "retrieval.passage" in rendered

    def test_jina_late_chunking_inactive(self) -> None:
        """When is_late_chunking=False, 'retrieval.passage' must not appear."""
        config = _make_config(
            framework="langchain",
            indexing=IndexingConfig(
                embedding=JinaEmbeddingConfig(late_chunking=False),
                vector_db=QdrantConfig(),
            ),
        )
        rendered = _loader().render_stage("embedding", "jina", _embedding_ctx(config))
        assert "retrieval.passage" not in rendered


# ─── Phase 5: VectorDB Stage Templates ───────────────────────────────────────


class TestEmbeddingDimensionLookup:
    def test_nomic_uses_configured_matryoshka_dimensionality(self) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=NomicEmbeddingConfig(dimensionality=256),
                vector_db=QdrantConfig(),
            )
        )
        assert _get_embedding_dim(config) == 256

    @pytest.mark.parametrize(
        ("embedding", "expected_dim"),
        [
            (GeminiEmbeddingConfig(model="text-embedding-004"), 768),
            (GeminiEmbeddingConfig(model="embedding-001"), 3072),
            (VoyageEmbeddingConfig(model="voyage-3-large"), 2048),
            (VoyageEmbeddingConfig(model="voyage-3"), 1024),
            (VoyageEmbeddingConfig(model="voyage-3-lite"), 512),
            (VoyageEmbeddingConfig(model="voyage-code-3"), 2048),
            (VoyageEmbeddingConfig(model="voyage-finance-2"), 1024),
            (VoyageEmbeddingConfig(model="voyage-law-2"), 1024),
        ],
    )
    def test_allowed_embedding_models_have_explicit_dimensions(
        self,
        embedding: object,
        expected_dim: int,
    ) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=embedding,  # type: ignore[arg-type]
                vector_db=QdrantConfig(),
            )
        )
        assert _get_embedding_dim(config) == expected_dim


def _vectordb_ctx(config: RAGPipelineConfig) -> dict:
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "vector_db": config.indexing.vector_db,
        "embedding_dim": _get_embedding_dim(config),
    }


class TestEmbeddingDimensions:
    @pytest.mark.parametrize(
        ("embedding", "expected_dim"),
        [
            (GeminiEmbeddingConfig(model="text-embedding-004"), 768),
            (GeminiEmbeddingConfig(model="embedding-001"), 3072),
            (VoyageEmbeddingConfig(model="voyage-3-large"), 2048),
            (VoyageEmbeddingConfig(model="voyage-3"), 1024),
            (VoyageEmbeddingConfig(model="voyage-3-lite"), 512),
            (VoyageEmbeddingConfig(model="voyage-code-3"), 2048),
        ],
    )
    def test_embedding_dim_lookup_matches_schema_literals(
        self,
        embedding: object,
        expected_dim: int,
    ) -> None:
        config = _make_config(
            indexing=IndexingConfig(
                embedding=embedding,  # type: ignore[arg-type]
                vector_db=ChromaDBConfig(),
            )
        )
        assert _get_embedding_dim(config) == expected_dim


class TestVectorDBStages:
    """
    Tests for stages/vectordb/:
      chromadb, qdrant, pinecone, weaviate, milvus, pgvector
    """

    # ── chromadb ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_chromadb_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=ChromaDBConfig()),
        )
        rendered = _loader().render_stage("vectordb", "chromadb", _vectordb_ctx(config))
        assert rendered.strip()
        assert "def build_vectordb(embedder" in rendered

    # ── qdrant ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_qdrant_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
        )
        rendered = _loader().render_stage("vectordb", "qdrant", _vectordb_ctx(config))
        assert rendered.strip()
        assert "def build_vectordb(embedder" in rendered

    # ── pinecone ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_pinecone_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=PineconeConfig()),
        )
        rendered = _loader().render_stage("vectordb", "pinecone", _vectordb_ctx(config))
        assert rendered.strip()
        assert "def build_vectordb(embedder" in rendered

    # ── weaviate ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_weaviate_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=WeaviateConfig()),
        )
        rendered = _loader().render_stage("vectordb", "weaviate", _vectordb_ctx(config))
        assert rendered.strip()
        assert "def build_vectordb(embedder" in rendered

    # ── milvus ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_milvus_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=MilvusConfig()),
        )
        rendered = _loader().render_stage("vectordb", "milvus", _vectordb_ctx(config))
        assert rendered.strip()
        assert "def build_vectordb(embedder" in rendered

    # ── pgvector ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_pgvector_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=PgVectorConfig()),
        )
        rendered = _loader().render_stage("vectordb", "pgvector", _vectordb_ctx(config))
        assert rendered.strip()
        assert "def build_vectordb(embedder" in rendered

    # ── specific assertions ───────────────────────────────────────────────────

    def test_llamaindex_milvus_uses_gemini_embedding_dim(self) -> None:
        config = _make_config(
            framework="llamaindex",
            indexing=IndexingConfig(
                embedding=GeminiEmbeddingConfig(model="text-embedding-004"),
                vector_db=MilvusConfig(),
            ),
        )
        rendered = _loader().render_stage("vectordb", "milvus", _vectordb_ctx(config))
        assert "dim=768" in rendered

    def test_llamaindex_pgvector_uses_voyage_embedding_dim(self) -> None:
        config = _make_config(
            framework="llamaindex",
            indexing=IndexingConfig(
                embedding=VoyageEmbeddingConfig(model="voyage-3-large"),
                vector_db=PgVectorConfig(),
            ),
        )
        rendered = _loader().render_stage("vectordb", "pgvector", _vectordb_ctx(config))
        assert "embed_dim=2048" in rendered

    def test_vectordb_build_fn_takes_embedder(self) -> None:
        """All vectordb templates must have 'def build_vectordb(embedder' signature."""
        config = _make_config(
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig())
        )
        rendered = _loader().render_stage("vectordb", "qdrant", _vectordb_ctx(config))
        assert "def build_vectordb(embedder" in rendered

    def test_qdrant_uses_collection_name(self) -> None:
        """QdrantConfig.collection_name must appear in the rendered output."""
        config = _make_config(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(collection_name="my_collection"),
            )
        )
        rendered = _loader().render_stage("vectordb", "qdrant", _vectordb_ctx(config))
        assert "my_collection" in rendered


# ─── Phase 6: Retrieval Stage Templates ──────────────────────────────────────


def _retrieval_ctx(config: RAGPipelineConfig) -> dict:
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "retrieval": config.retrieval,
        "vector_db_type": config.indexing.vector_db.type,
    }


class TestRetrievalStages:
    """
    Tests for stages/retrieval/:
      dense, hybrid_rrf, hybrid_weighted, small_to_big, sentence_window
    """

    # ── dense ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_dense_renders(self, framework: str) -> None:
        config = _make_config(framework=framework, retrieval=DenseRetrievalConfig())
        rendered = _loader().render_stage("retrieval", "dense", _retrieval_ctx(config))
        assert rendered.strip()
        assert "def build_retrieval(vectorstore" in rendered

    # ── hybrid_rrf ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_hybrid_rrf_renders(self, framework: str) -> None:
        config = _make_config(framework=framework, retrieval=HybridRRFConfig())
        rendered = _loader().render_stage("retrieval", "hybrid_rrf", _retrieval_ctx(config))
        assert rendered.strip()
        assert "def build_retrieval(vectorstore" in rendered

    # ── hybrid_weighted ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_hybrid_weighted_renders(self, framework: str) -> None:
        config = _make_config(framework=framework, retrieval=HybridWeightedConfig())
        rendered = _loader().render_stage("retrieval", "hybrid_weighted", _retrieval_ctx(config))
        assert rendered.strip()
        assert "def build_retrieval(vectorstore" in rendered

    # ── small_to_big ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_small_to_big_renders(self, framework: str) -> None:
        config = _make_config(framework=framework, retrieval=SmallToBigConfig())
        rendered = _loader().render_stage("retrieval", "small_to_big", _retrieval_ctx(config))
        assert rendered.strip()
        assert "def build_retrieval(vectorstore" in rendered

    # ── sentence_window ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_sentence_window_renders(self, framework: str) -> None:
        config = _make_config(framework=framework, retrieval=SentenceWindowConfig())
        rendered = _loader().render_stage("retrieval", "sentence_window", _retrieval_ctx(config))
        assert rendered.strip()
        assert "def build_retrieval(vectorstore" in rendered

    # ── specific assertions ───────────────────────────────────────────────────

    def test_hybrid_rrf_k_interpolated(self) -> None:
        """rrf_k config value must appear in the rendered hybrid_rrf output."""
        config = _make_config(framework="langchain", retrieval=HybridRRFConfig(rrf_k=42))
        rendered = _loader().render_stage("retrieval", "hybrid_rrf", _retrieval_ctx(config))
        assert "42" in rendered

    def test_hybrid_weighted_alpha_interpolated(self) -> None:
        """alpha config value must appear in the rendered hybrid_weighted output."""
        config = _make_config(framework="langchain", retrieval=HybridWeightedConfig(alpha=0.7))
        rendered = _loader().render_stage("retrieval", "hybrid_weighted", _retrieval_ctx(config))
        assert "0.7" in rendered

    def test_sentence_window_langchain_approximation_comment(self) -> None:
        """LangChain sentence_window must contain the approximation comment."""
        config = _make_config(framework="langchain", retrieval=SentenceWindowConfig())
        rendered = _loader().render_stage("retrieval", "sentence_window", _retrieval_ctx(config))
        assert "LangChain approximation" in rendered

    def test_small_to_big_parent_chunk_size(self) -> None:
        """parent_chunk_size must appear in the rendered output."""
        config = _make_config(
            framework="langchain",
            retrieval=SmallToBigConfig(child_chunk_size=128, parent_chunk_size=512),
        )
        rendered = _loader().render_stage("retrieval", "small_to_big", _retrieval_ctx(config))
        assert "512" in rendered

    def test_dense_langchain_uses_as_retriever(self) -> None:
        """.as_retriever() must appear in the LangChain dense output."""
        config = _make_config(framework="langchain", retrieval=DenseRetrievalConfig())
        rendered = _loader().render_stage("retrieval", "dense", _retrieval_ctx(config))
        assert ".as_retriever(" in rendered


# ─── Phase 7: Reranker Stage Templates ───────────────────────────────────────


def _reranker_ctx(config: RAGPipelineConfig) -> dict:
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "reranker": config.post_retrieval.reranker,
    }


class TestRerankerStages:
    """
    Tests for stages/reranker/:
      cohere, cross_encoder, colbert, flashrank
    """

    # ── cohere ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_cohere_reranker_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()),
        )
        rendered = _loader().render_stage("reranker", "cohere", _reranker_ctx(config))
        assert rendered.strip()
        if framework == "langchain":
            assert "def build_reranker(retriever" in rendered
        else:
            assert "def build_reranker()" in rendered

    # ── cross_encoder ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_cross_encoder_reranker_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            post_retrieval=PostRetrievalConfig(reranker=CrossEncoderRerankerConfig()),
        )
        rendered = _loader().render_stage("reranker", "cross_encoder", _reranker_ctx(config))
        assert rendered.strip()
        if framework == "langchain":
            assert "def build_reranker(retriever" in rendered
        else:
            assert "def build_reranker()" in rendered

    # ── colbert ───────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_colbert_reranker_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            post_retrieval=PostRetrievalConfig(reranker=ColBERTRerankerConfig()),
        )
        rendered = _loader().render_stage("reranker", "colbert", _reranker_ctx(config))
        assert rendered.strip()
        if framework == "langchain":
            assert "def build_reranker(retriever" in rendered
        else:
            assert "def build_reranker()" in rendered

    # ── flashrank ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_flashrank_reranker_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework,
            post_retrieval=PostRetrievalConfig(reranker=FlashRankRerankerConfig()),
        )
        rendered = _loader().render_stage("reranker", "flashrank", _reranker_ctx(config))
        assert rendered.strip()
        if framework == "langchain":
            assert "def build_reranker(retriever" in rendered
        else:
            assert "def build_reranker()" in rendered

    # ── specific assertions ───────────────────────────────────────────────────

    def test_reranker_build_fn_wraps_retriever(self) -> None:
        """All reranker templates must accept 'retriever' as argument."""
        config = _make_config(
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()),
        )
        rendered = _loader().render_stage("reranker", "cohere", _reranker_ctx(config))
        assert "def build_reranker(retriever" in rendered

    def test_cohere_reranker_top_n_interpolated(self) -> None:
        """top_n config value must appear in the rendered cohere reranker output."""
        config = _make_config(
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig(top_n=7)),
        )
        rendered = _loader().render_stage("reranker", "cohere", _reranker_ctx(config))
        assert "7" in rendered


# ─── Phase 8: LLM Stage Templates ────────────────────────────────────────────


def _llm_ctx(config: RAGPipelineConfig) -> dict:
    return {
        "config": config,
        "framework": str(config.framework),
        "pipeline_name": config.name,
        "dependencies": get_dependencies(config),
        "python_version": "3.11",
        "llm": config.generation.llm,
        "prompt_template": config.generation.prompt_template,
        "advanced": config.generation.advanced,
    }


class TestLLMStages:
    """
    Tests for stages/llm/:
      openai, anthropic, cohere_llm, ollama
    """

    # ── openai ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_openai_llm_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework, generation=GenerationConfig(llm=OpenAILLMConfig())
        )
        rendered = _loader().render_stage("llm", "openai", _llm_ctx(config))
        assert rendered.strip()
        assert "def build_llm()" in rendered

    # ── anthropic ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_anthropic_llm_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework, generation=GenerationConfig(llm=AnthropicLLMConfig())
        )
        rendered = _loader().render_stage("llm", "anthropic", _llm_ctx(config))
        assert rendered.strip()
        assert "def build_llm()" in rendered

    # ── cohere_llm ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_cohere_llm_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework, generation=GenerationConfig(llm=CohereLLMConfig())
        )
        rendered = _loader().render_stage("llm", "cohere_llm", _llm_ctx(config))
        assert rendered.strip()
        assert "def build_llm()" in rendered

    # ── ollama ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
    def test_ollama_llm_renders(self, framework: str) -> None:
        config = _make_config(
            framework=framework, generation=GenerationConfig(llm=OllamaLLMConfig())
        )
        rendered = _loader().render_stage("llm", "ollama", _llm_ctx(config))
        assert rendered.strip()
        assert "def build_llm()" in rendered

    # ── specific assertions ───────────────────────────────────────────────────

    def test_llm_temperature_interpolated(self) -> None:
        """temperature config value must appear in the rendered LLM output."""
        config = _make_config(generation=GenerationConfig(llm=OpenAILLMConfig(temperature=0.3)))
        rendered = _loader().render_stage("llm", "openai", _llm_ctx(config))
        assert "0.3" in rendered

    def test_ollama_base_url_interpolated(self) -> None:
        """base_url must appear in the rendered Ollama output."""
        config = _make_config(
            generation=GenerationConfig(llm=OllamaLLMConfig(base_url="http://custom:11434"))
        )
        rendered = _loader().render_stage("llm", "ollama", _llm_ctx(config))
        assert "http://custom:11434" in rendered

    def test_ollama_model_name_interpolated(self) -> None:
        """model name must appear in the rendered Ollama output."""
        config = _make_config(generation=GenerationConfig(llm=OllamaLLMConfig(model="mistral")))
        rendered = _loader().render_stage("llm", "ollama", _llm_ctx(config))
        assert "mistral" in rendered

    def test_openai_llm_max_tokens_interpolated(self) -> None:
        """max_tokens must appear in the rendered OpenAI LLM output."""
        config = _make_config(generation=GenerationConfig(llm=OpenAILLMConfig(max_tokens=4096)))
        rendered = _loader().render_stage("llm", "openai", _llm_ctx(config))
        assert "4096" in rendered


# ─── Phase 9: End-to-End + Cross-Cutting Tests ───────────────────────────────


def _e2e_result(framework: str, **kwargs: Any) -> GeneratorResult:
    """Build a minimal valid config, call generate(), return the result."""
    defaults: dict[str, Any] = {
        "name": "e2e-test",
        "framework": framework,
        "indexing": IndexingConfig(
            embedding=OpenAIEmbeddingConfig(),
            vector_db=QdrantConfig(),
        ),
        "generation": GenerationConfig(llm=OpenAILLMConfig()),
    }
    defaults.update(kwargs)
    config = RAGPipelineConfig(**defaults)  # type: ignore[arg-type]
    return generate(config)


class TestEndToEnd:
    """
    Full generate() smoke tests — verify validation_passed=True, correct file set,
    no errors, and key content across 8 representative configurations.
    """

    _EXPECTED_FILES = {
        "pipeline.py",
        "ingestion.py",
        "pyproject.toml",
        ".env.example",
        "README.md",
        "Dockerfile",
    }

    def _assert_valid(self, result: GeneratorResult) -> None:
        assert result.validation_passed, f"Validation failed: {result.errors}"
        assert not result.errors
        for key in self._EXPECTED_FILES:
            assert key in result.files, f"Missing file: {key}"

    def test_langchain_openai_qdrant_hybrid_rrf_cohere_reranker_gpt4o(self) -> None:
        result = _e2e_result(
            "langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            retrieval=HybridRRFConfig(),
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig(model="gpt-4o")),
        )
        self._assert_valid(result)
        assert "docker-compose.yml" in result.files

    def test_langchain_cohere_pinecone_dense_no_reranker_anthropic(self) -> None:
        result = _e2e_result(
            "langchain",
            indexing=IndexingConfig(embedding=CohereEmbeddingConfig(), vector_db=PineconeConfig()),
            retrieval=DenseRetrievalConfig(),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        )
        self._assert_valid(result)
        assert "docker-compose.yml" not in result.files  # Pinecone = managed, no compose

    def test_langchain_bge_m3_weaviate_hybrid_weighted_cross_encoder_ollama(self) -> None:
        result = _e2e_result(
            "langchain",
            indexing=IndexingConfig(embedding=BGEM3EmbeddingConfig(), vector_db=WeaviateConfig()),
            retrieval=HybridWeightedConfig(),
            post_retrieval=PostRetrievalConfig(reranker=CrossEncoderRerankerConfig()),
            generation=GenerationConfig(llm=OllamaLLMConfig()),
        )
        self._assert_valid(result)

    def test_langchain_jina_chromadb_sentence_window_no_reranker_gpt4o_mini(self) -> None:
        result = _e2e_result(
            "langchain",
            indexing=IndexingConfig(embedding=JinaEmbeddingConfig(), vector_db=ChromaDBConfig()),
            retrieval=SentenceWindowConfig(),
            generation=GenerationConfig(llm=OpenAILLMConfig(model="gpt-4o-mini")),
        )
        self._assert_valid(result)

    def test_llamaindex_openai_qdrant_hybrid_rrf_cohere_reranker_gpt4o(self) -> None:
        result = _e2e_result(
            "llamaindex",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            retrieval=HybridRRFConfig(),
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig(model="gpt-4o")),
        )
        self._assert_valid(result)

    def test_llamaindex_voyage_milvus_small_to_big_colbert_anthropic(self) -> None:
        result = _e2e_result(
            "llamaindex",
            indexing=IndexingConfig(embedding=VoyageEmbeddingConfig(), vector_db=MilvusConfig()),
            retrieval=SmallToBigConfig(),
            post_retrieval=PostRetrievalConfig(reranker=ColBERTRerankerConfig()),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        )
        self._assert_valid(result)

    def test_llamaindex_nomic_pgvector_dense_flashrank_cohere_llm(self) -> None:
        result = _e2e_result(
            "llamaindex",
            indexing=IndexingConfig(embedding=NomicEmbeddingConfig(), vector_db=PgVectorConfig()),
            retrieval=DenseRetrievalConfig(),
            post_retrieval=PostRetrievalConfig(reranker=FlashRankRerankerConfig()),
            generation=GenerationConfig(llm=CohereLLMConfig()),
        )
        self._assert_valid(result)

    def test_llamaindex_jina_late_chromadb_dense_no_reranker_ollama(self) -> None:
        result = _e2e_result(
            "llamaindex",
            indexing=IndexingConfig(
                embedding=JinaEmbeddingConfig(late_chunking=True),
                vector_db=ChromaDBConfig(),
                chunking=LateChunkingConfig(),
            ),
            retrieval=DenseRetrievalConfig(),
            generation=GenerationConfig(llm=OllamaLLMConfig()),
        )
        self._assert_valid(result)


class TestEnvVarGeneration:
    """
    Verify .env.example contains exactly the right API keys for each config.
    """

    def _env_example(self, **kwargs: Any) -> str:
        config = RAGPipelineConfig(
            **{
                "name": "env-test",
                "framework": "langchain",
                "indexing": IndexingConfig(
                    embedding=OpenAIEmbeddingConfig(), vector_db=ChromaDBConfig()
                ),
                "generation": GenerationConfig(llm=OllamaLLMConfig()),
                **kwargs,
            }
        )  # type: ignore[arg-type]
        result = generate(config)
        return result.files[".env.example"]

    def test_openai_embedding_has_openai_key(self) -> None:
        """OpenAI embedding → OPENAI_API_KEY in .env.example."""
        env = self._env_example(
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=ChromaDBConfig())
        )
        assert "OPENAI_API_KEY" in env

    def test_pinecone_has_pinecone_key(self) -> None:
        """Pinecone vector DB → PINECONE_API_KEY in .env.example."""
        env = self._env_example(
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=PineconeConfig())
        )
        assert "PINECONE_API_KEY" in env

    def test_anthropic_llm_has_anthropic_key(self) -> None:
        """Anthropic LLM → ANTHROPIC_API_KEY in .env.example."""
        env = self._env_example(generation=GenerationConfig(llm=AnthropicLLMConfig()))
        assert "ANTHROPIC_API_KEY" in env

    def test_anthropic_not_duplicated_with_contextual_chunking(self) -> None:
        """Contextual chunking with claude + Anthropic LLM → only one ANTHROPIC_API_KEY entry."""
        config = RAGPipelineConfig(
            name="dedup-test",
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=ContextualChunkingConfig(context_model="claude-3-haiku-20240307"),
            ),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        )
        result = generate(config)
        env = result.files[".env.example"]
        # The key should appear at most once as a variable name
        assert env.count("ANTHROPIC_API_KEY=") == 1

    def test_ollama_has_no_api_key_vars(self) -> None:
        """Ollama LLM + ChromaDB + BGE-M3 → no API key vars (all local)."""
        config = RAGPipelineConfig(
            name="local-test",
            framework="langchain",
            indexing=IndexingConfig(embedding=BGEM3EmbeddingConfig(), vector_db=ChromaDBConfig()),
            generation=GenerationConfig(llm=OllamaLLMConfig()),
        )
        result = generate(config)
        env = result.files[".env.example"]
        # env.example should be essentially empty (no API key assignments)
        assert "API_KEY=" not in env

    def test_crag_tavily_is_blocked_before_env_generation(self) -> None:
        """CRAG is schema-parsed but blocked before env template generation."""
        config = RAGPipelineConfig(
            name="crag-test",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(enabled=True, web_search_provider="tavily")
                ),
            ),
        )
        result = generate(config)
        assert result.validation_passed is False
        assert ".env.example" not in result.files
        assert any("generation.advanced.crag" in error for error in result.errors)

    def test_contextual_mistral_does_not_scaffold_manual_provider_key(self) -> None:
        config = RAGPipelineConfig(
            name="ctx-mistral-test",
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=ContextualChunkingConfig(context_model="mistral-large"),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = generate(config)
        env = result.files[".env.example"]
        assert "OPENAI_API_KEY" in env
        assert "MISTRAL_API_KEY" not in env


class TestDependencyAccuracy:
    """
    Verify pyproject.toml dependencies match get_dependencies() output exactly.
    """

    def _pyproject(self, config: RAGPipelineConfig) -> str:
        return generate(config).files["pyproject.toml"]

    def test_langchain_openai_has_langchain_openai_dep(self) -> None:
        """LangChain + OpenAI → langchain-openai in pyproject.toml."""
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        assert "langchain-openai" in self._pyproject(config)

    def test_llamaindex_cohere_has_cohere_embedding_dep(self) -> None:
        """LlamaIndex + Cohere embedding → llama-index-embeddings-cohere in pyproject.toml."""
        config = RAGPipelineConfig(
            name="t",
            framework="llamaindex",
            indexing=IndexingConfig(embedding=CohereEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        assert "llama-index-embeddings-cohere" in self._pyproject(config)

    def test_bge_m3_has_flagembedding_dep(self) -> None:
        """BGE-M3 embedding (any framework) → FlagEmbedding in pyproject.toml."""
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=BGEM3EmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        assert "FlagEmbedding" in self._pyproject(config)

    def test_semantic_chunking_has_sentence_transformers_dep(self) -> None:
        """Semantic chunking → sentence-transformers in pyproject.toml."""
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
                chunking=SemanticChunkingConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        assert "sentence-transformers" in self._pyproject(config)

    def test_crag_duckduckgo_does_not_advertise_duckduckgo_dep(self) -> None:
        """Unsupported CRAG must not advertise duckduckgo runtime dependencies."""
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(enabled=True, web_search_provider="duckduckgo")
                ),
            ),
        )
        assert "duckduckgo-search" not in get_dependencies(config)

    def test_ragas_evaluation_has_ragas_dep(self) -> None:
        """Ragas evaluation framework → ragas in pyproject.toml."""
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
            evaluation=EvaluationConfig(framework="ragas"),
        )
        assert "ragas" in self._pyproject(config)


# ─── Phase 10: Structural Hardening ──────────────────────────────────────────


class TestImportReachability:
    """Package-level import sanity checks."""

    def test_package_importable(self) -> None:
        """import ragfactory must succeed."""
        import ragfactory  # noqa: F401

    def test_generate_importable(self) -> None:
        """from ragfactory.core.generator import generate must succeed."""
        from ragfactory.core.generator import generate as _gen  # noqa: F401

    def test_templates_dir_exists(self) -> None:
        """The shipped templates directory must exist on disk."""
        assert _DEFAULT_TEMPLATE_DIR.exists()
        assert _DEFAULT_TEMPLATE_DIR.is_dir()

    def test_all_42_templates_present(self) -> None:
        """Every expected template file must exist in the templates directory."""
        expected = [
            # Entrypoints
            "entrypoints/langchain/pipeline.py.j2",
            "entrypoints/langchain/ingestion.py.j2",
            "entrypoints/llamaindex/pipeline.py.j2",
            "entrypoints/llamaindex/ingestion.py.j2",
            "entrypoints/common/pyproject.toml.j2",
            "entrypoints/common/.env.example.j2",
            "entrypoints/common/README.md.j2",
            "entrypoints/common/Dockerfile.j2",
            "entrypoints/common/docker-compose.yml.j2",
            # Chunking (7)
            "stages/chunking/fixed.py.j2",
            "stages/chunking/recursive.py.j2",
            "stages/chunking/semantic.py.j2",
            "stages/chunking/contextual.py.j2",
            "stages/chunking/late.py.j2",
            "stages/chunking/page_level.py.j2",
            "stages/chunking/proposition.py.j2",
            # Embedding (7)
            "stages/embedding/openai.py.j2",
            "stages/embedding/cohere.py.j2",
            "stages/embedding/voyage.py.j2",
            "stages/embedding/gemini.py.j2",
            "stages/embedding/bge_m3.py.j2",
            "stages/embedding/nomic.py.j2",
            "stages/embedding/jina.py.j2",
            # VectorDB (6)
            "stages/vectordb/chromadb.py.j2",
            "stages/vectordb/qdrant.py.j2",
            "stages/vectordb/pinecone.py.j2",
            "stages/vectordb/weaviate.py.j2",
            "stages/vectordb/milvus.py.j2",
            "stages/vectordb/pgvector.py.j2",
            # Retrieval (5)
            "stages/retrieval/dense.py.j2",
            "stages/retrieval/hybrid_rrf.py.j2",
            "stages/retrieval/hybrid_weighted.py.j2",
            "stages/retrieval/small_to_big.py.j2",
            "stages/retrieval/sentence_window.py.j2",
            # Reranker (4)
            "stages/reranker/cohere.py.j2",
            "stages/reranker/cross_encoder.py.j2",
            "stages/reranker/colbert.py.j2",
            "stages/reranker/flashrank.py.j2",
            # LLM (4)
            "stages/llm/openai.py.j2",
            "stages/llm/anthropic.py.j2",
            "stages/llm/cohere_llm.py.j2",
            "stages/llm/ollama.py.j2",
        ]
        for rel_path in expected:
            full_path = _DEFAULT_TEMPLATE_DIR / rel_path
            assert full_path.exists(), f"Missing template: {rel_path}"


class TestASTValidation:
    """
    Belt-and-suspenders: verify that generate() produces syntactically valid Python
    for every smoke E2E scenario (the generator already runs ast.parse, but these
    tests make failures explicit in the test report).
    """

    def _assert_python_valid(self, config: RAGPipelineConfig) -> None:
        result = generate(config)
        assert result.validation_passed, f"AST validation failed for {config.name}: {result.errors}"
        assert result.errors == []

    def test_pipeline_py_valid_langchain_full(self) -> None:
        config = RAGPipelineConfig(
            name="ast-lc",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            retrieval=HybridRRFConfig(),
            pre_retrieval=PreRetrievalConfig(query_rewriting=QueryRewritingConfig(enabled=True)),
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        self._assert_python_valid(config)

    def test_pipeline_py_valid_llamaindex_full(self) -> None:
        config = RAGPipelineConfig(
            name="ast-li",
            framework="llamaindex",
            indexing=IndexingConfig(
                embedding=JinaEmbeddingConfig(late_chunking=True), vector_db=ChromaDBConfig()
            ),
            retrieval=SentenceWindowConfig(),
            pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=True)),
            post_retrieval=PostRetrievalConfig(reranker=CrossEncoderRerankerConfig()),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        )
        self._assert_python_valid(config)

    def test_ingestion_py_valid_langchain(self) -> None:
        config = RAGPipelineConfig(
            name="ast-ingest-lc",
            framework="langchain",
            indexing=IndexingConfig(embedding=CohereEmbeddingConfig(), vector_db=PineconeConfig()),
            generation=GenerationConfig(llm=OllamaLLMConfig()),
        )
        self._assert_python_valid(config)

    def test_ingestion_py_valid_llamaindex(self) -> None:
        config = RAGPipelineConfig(
            name="ast-ingest-li",
            framework="llamaindex",
            indexing=IndexingConfig(embedding=NomicEmbeddingConfig(), vector_db=MilvusConfig()),
            generation=GenerationConfig(llm=CohereLLMConfig()),
        )
        self._assert_python_valid(config)


class TestDockerComposeAbsence:
    """
    Regression tests: docker-compose.yml is NEVER generated for
    managed/local-only vector DBs.
    """

    def test_docker_compose_absent_for_chromadb_regression(self) -> None:
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=ChromaDBConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = generate(config)
        assert "docker-compose.yml" not in result.files

    def test_docker_compose_absent_for_pinecone_regression(self) -> None:
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=PineconeConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = generate(config)
        assert "docker-compose.yml" not in result.files

    def test_docker_compose_present_for_qdrant_regression(self) -> None:
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = generate(config)
        assert "docker-compose.yml" in result.files

    def test_docker_compose_present_for_pgvector_regression(self) -> None:
        config = RAGPipelineConfig(
            name="t",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=PgVectorConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = generate(config)
        assert "docker-compose.yml" in result.files


class TestStrictUndefined:
    """
    Verify no template variable is silently missing: all stage/entrypoint
    templates render without UndefinedError for representative configs.
    """

    def test_all_stages_render_without_undefined_error(self) -> None:
        """Full generate() on a maximally-featured config must not raise GeneratorError."""
        config = RAGPipelineConfig(
            name="strict-test",
            framework="langchain",
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(),
                embedding=JinaEmbeddingConfig(late_chunking=True),
                vector_db=QdrantConfig(),
            ),
            retrieval=HybridRRFConfig(),
            pre_retrieval=PreRetrievalConfig(
                query_rewriting=QueryRewritingConfig(enabled=True),
            ),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(),
            ),
            generation=GenerationConfig(
                llm=AnthropicLLMConfig(),
            ),
        )
        # Should not raise — returns a result with all templates rendered.
        result = generate(config)
        assert result.validation_passed, f"Unexpected errors: {result.errors}"

    def test_llamaindex_all_stages_render(self) -> None:
        """Same as above for LlamaIndex framework."""
        config = RAGPipelineConfig(
            name="strict-li",
            framework="llamaindex",
            indexing=IndexingConfig(
                chunking=SemanticChunkingConfig(),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=WeaviateConfig(),
            ),
            retrieval=SmallToBigConfig(),
            pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=True)),
            post_retrieval=PostRetrievalConfig(reranker=FlashRankRerankerConfig()),
            generation=GenerationConfig(llm=OllamaLLMConfig()),
        )
        result = generate(config)
        assert result.validation_passed, f"Unexpected errors: {result.errors}"


class TestConfigYamlRoundTrip:
    """
    Round-trip: config → YAML string → back to config → generate() → same output.
    """

    def _roundtrip(self, config: RAGPipelineConfig) -> GeneratorResult:
        yaml_str = config.to_yaml()
        config2 = RAGPipelineConfig.from_yaml(yaml_str)
        return generate(config2)

    def test_roundtrip_langchain_openai_qdrant(self) -> None:
        config = RAGPipelineConfig(
            name="rt-1",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = self._roundtrip(config)
        assert result.validation_passed

    def test_roundtrip_llamaindex_cohere_pinecone(self) -> None:
        config = RAGPipelineConfig(
            name="rt-2",
            framework="llamaindex",
            indexing=IndexingConfig(embedding=CohereEmbeddingConfig(), vector_db=PineconeConfig()),
            generation=GenerationConfig(llm=CohereLLMConfig()),
        )
        result = self._roundtrip(config)
        assert result.validation_passed

    def test_roundtrip_preserves_pipeline_name(self) -> None:
        config = RAGPipelineConfig(
            name="my-unique-pipeline",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = self._roundtrip(config)
        assert "my-unique-pipeline" in result.files["pipeline.py"]

    def test_roundtrip_with_reranker(self) -> None:
        config = RAGPipelineConfig(
            name="rt-reranker",
            framework="langchain",
            indexing=IndexingConfig(embedding=OpenAIEmbeddingConfig(), vector_db=QdrantConfig()),
            post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        )
        result = self._roundtrip(config)
        assert result.validation_passed
        assert "build_reranker" in result.files["pipeline.py"]
