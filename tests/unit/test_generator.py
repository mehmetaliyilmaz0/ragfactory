"""
Unit tests for ragfactory.core.generator.

All tests use stub templates from tests/fixtures/stub_templates/
so they test the generator ENGINE, not template content.
Template content testing is Phase 1c's test_templates.py.

Coverage:
  1. Return type and structure
  2. All expected files present
  3. Python files pass ast.parse()
  4. Pure function — no disk writes
  5. Framework routing (langchain vs llamaindex)
  6. docker-compose.yml conditional generation
  7. Reranker stage skipped when None
  8. config_yaml round-trip
  9. Missing template → error result (not exception)
  10. StrictUndefined catches undefined variables
  11. Syntax error in generated Python → error in result
  12. All optional fields None — no crash
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragfactory.core.config import (
    AdvancedGenerationConfig,
    AgenticConfig,
    AnthropicLLMConfig,
    BGEM3EmbeddingConfig,
    ChromaDBConfig,
    CohereRerankerConfig,
    CRAGConfig,
    DenseRetrievalConfig,
    FLAREConfig,
    Framework,
    GenerationConfig,
    HybridRRFConfig,
    IndexingConfig,
    JinaEmbeddingConfig,
    LateChunkingConfig,
    MilvusConfig,
    OllamaLLMConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    PgVectorConfig,
    PostRetrievalConfig,
    QdrantConfig,
    RAGPipelineConfig,
    RecursiveChunkingConfig,
    SentenceWindowConfig,
    VoyageEmbeddingConfig,
    WeaviateConfig,
)
from ragfactory.core.generator import (
    GeneratedFile,
    GeneratorError,
    GeneratorResult,
    TemplateLoader,
    generate,
    _validate_python,
)

# Path to stub templates used for all engine tests
STUB_TEMPLATE_DIR = Path(__file__).parent.parent / "fixtures" / "stub_templates"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _cfg(**overrides: object) -> RAGPipelineConfig:
    defaults: dict[str, object] = {
        "name": "test-pipeline",
        "indexing": IndexingConfig(
            embedding=OpenAIEmbeddingConfig(),
            vector_db=QdrantConfig(),
        ),
        "generation": GenerationConfig(llm=OpenAILLMConfig()),
    }
    defaults.update(overrides)
    return RAGPipelineConfig(**defaults)  # type: ignore[arg-type]


def _gen(config: RAGPipelineConfig | None = None) -> GeneratorResult:
    return generate(config or _cfg(), template_dir=STUB_TEMPLATE_DIR)


# ─── 1. Return type and structure ─────────────────────────────────────────────


class TestReturnTypeAndStructure:
    def test_returns_generator_result(self) -> None:
        result = _gen()
        assert isinstance(result, GeneratorResult)

    def test_files_is_dict(self) -> None:
        result = _gen()
        assert isinstance(result.files, dict)

    def test_generated_files_is_list(self) -> None:
        result = _gen()
        assert isinstance(result.generated_files, list)
        assert all(isinstance(f, GeneratedFile) for f in result.generated_files)

    def test_validation_passed_is_bool(self) -> None:
        result = _gen()
        assert isinstance(result.validation_passed, bool)

    def test_errors_is_list(self) -> None:
        result = _gen()
        assert isinstance(result.errors, list)

    def test_config_yaml_is_string(self) -> None:
        result = _gen()
        assert isinstance(result.config_yaml, str)
        assert len(result.config_yaml) > 0


# ─── 2. Expected files present ────────────────────────────────────────────────


class TestExpectedFiles:
    _ALWAYS_PRESENT = {
        "pipeline.py",
        "ingestion.py",
        "pyproject.toml",
        ".env.example",
        "README.md",
        "Dockerfile",
    }

    def test_all_required_files_generated(self) -> None:
        result = _gen()
        assert self._ALWAYS_PRESENT.issubset(result.files.keys()), (
            f"Missing files: {self._ALWAYS_PRESENT - result.files.keys()}"
        )

    def test_files_dict_matches_generated_files_list(self) -> None:
        result = _gen()
        list_paths = {f.path for f in result.generated_files}
        dict_paths = set(result.files.keys())
        assert list_paths == dict_paths

    def test_generated_files_have_content(self) -> None:
        result = _gen()
        for f in result.generated_files:
            assert f.content, f"Empty content for {f.path}"


# ─── 3. Python files pass ast.parse() ────────────────────────────────────────


class TestAstValidation:
    def test_all_python_files_pass_ast_parse(self) -> None:
        result = _gen()
        assert result.validation_passed is True
        assert result.errors == []

    def test_prompt_injection_safety(self) -> None:
        malicious_prompt = 'You are an assistant. {context} {question}\n"; import os; os.system("echo INJECTED"); #'
        cfg = _cfg(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                prompt_template=malicious_prompt
            )
        )
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert result.validation_passed is True
        # Verify it is safely escaped as a python string representation
        assert "echo INJECTED" in result.files["pipeline.py"]

    def test_python_files_flagged_correctly(self) -> None:
        result = _gen()
        python_files = [f for f in result.generated_files if f.is_python]
        non_python   = [f for f in result.generated_files if not f.is_python]

        python_paths = {f.path for f in python_files}
        non_py_paths = {f.path for f in non_python}

        assert "pipeline.py"  in python_paths
        assert "ingestion.py" in python_paths
        assert "pyproject.toml" in non_py_paths
        assert "README.md"      in non_py_paths
        assert "Dockerfile"     in non_py_paths

    def test_validate_python_valid_code(self) -> None:
        errors = _validate_python("x = 1\ny = 2\n", "test.py")
        assert errors == []

    def test_validate_python_invalid_code(self) -> None:
        errors = _validate_python("def broken(\n", "bad.py")
        assert len(errors) == 1
        assert "bad.py" in errors[0]


# ─── 4. Pure function — no disk writes ────────────────────────────────────────


class TestPureFunction:
    def test_no_files_written_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        _gen()
        written = list(tmp_path.iterdir())
        assert written == [], f"Generator wrote files to disk: {written}"

    def test_called_twice_same_result(self) -> None:
        cfg = _cfg()
        r1 = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        r2 = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert r1.files == r2.files


# ─── 5. Framework routing ─────────────────────────────────────────────────────


class TestFrameworkRouting:
    def test_langchain_generates_langchain_pipeline(self) -> None:
        result = _gen(_cfg(framework=Framework.LANGCHAIN))
        assert result.validation_passed is True
        assert "pipeline.py" in result.files
        # Stub template marks langchain
        assert "langchain" in result.files["pipeline.py"]

    def test_llamaindex_generates_llamaindex_pipeline(self) -> None:
        result = _gen(_cfg(framework=Framework.LLAMAINDEX))
        assert result.validation_passed is True
        assert "llamaindex" in result.files["pipeline.py"]

    def test_langchain_and_llamaindex_produce_different_content(self) -> None:
        r_lc  = _gen(_cfg(framework=Framework.LANGCHAIN))
        r_lli = _gen(_cfg(framework=Framework.LLAMAINDEX))
        assert r_lc.files["pipeline.py"] != r_lli.files["pipeline.py"]


# ─── 6. docker-compose conditional generation ────────────────────────────────


class TestDockerComposeConditional:
    def test_qdrant_includes_docker_compose(self) -> None:
        result = _gen(_cfg(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert "docker-compose.yml" in result.files

    def test_weaviate_includes_docker_compose(self) -> None:
        result = _gen(_cfg(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=WeaviateConfig(),
            ),
        ))
        assert "docker-compose.yml" in result.files

    def test_milvus_includes_docker_compose(self) -> None:
        result = _gen(_cfg(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=MilvusConfig(),
            ),
        ))
        assert "docker-compose.yml" in result.files

    def test_pgvector_includes_docker_compose(self) -> None:
        result = _gen(_cfg(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=PgVectorConfig(),
            ),
        ))
        assert "docker-compose.yml" in result.files

    def test_chromadb_no_docker_compose(self) -> None:
        result = _gen(_cfg(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
        ))
        assert "docker-compose.yml" not in result.files


# ─── 7. Reranker optional ─────────────────────────────────────────────────────


class TestRerankerOptional:
    def test_no_reranker_no_reranker_stage(self) -> None:
        result = _gen(_cfg(
            post_retrieval=PostRetrievalConfig(reranker=None),
        ))
        # reranker key should not be in the rendered stages embedded in pipeline
        # (indirectly: pipeline.py should not contain "RERANKER_TYPE")
        assert result.validation_passed is True

    def test_with_reranker_pipeline_contains_reranker_content(self) -> None:
        result = _gen(_cfg(
            retrieval=DenseRetrievalConfig(top_k=20),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(top_n=5),
            ),
        ))
        assert result.validation_passed is True
        # Stub reranker template emits RERANKER_TYPE
        assert "RERANKER_TYPE" in result.files["pipeline.py"]


# ─── 8. config_yaml round-trip ────────────────────────────────────────────────


class TestConfigYamlRoundTrip:
    def test_config_yaml_matches_config(self) -> None:
        cfg = _cfg()
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        restored = RAGPipelineConfig.from_yaml(result.config_yaml)
        assert restored == cfg

    def test_config_yaml_present_even_on_error(self) -> None:
        # Use a non-existent template dir to force an error
        cfg = _cfg()
        result = generate(cfg, template_dir=Path("/nonexistent/path"))
        assert result.config_yaml == cfg.to_yaml()

    def test_config_yaml_contains_pipeline_name(self) -> None:
        cfg = _cfg(name="my-special-rag")
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert "my-special-rag" in result.config_yaml


# ─── 9. Missing template → error result ──────────────────────────────────────


class TestMissingTemplate:
    def test_missing_template_dir_returns_error_result(self) -> None:
        result = generate(_cfg(), template_dir=Path("/nonexistent/templates"))
        assert result.validation_passed is False
        assert len(result.errors) > 0
        assert result.files == {}

    def test_missing_template_does_not_raise(self) -> None:
        # generate() must never raise — always return GeneratorResult
        result = generate(_cfg(), template_dir=Path("/nonexistent"))
        assert isinstance(result, GeneratorResult)


class TestUnsupportedAdvancedGeneration:
    @pytest.mark.parametrize(
        "advanced",
        [
            AdvancedGenerationConfig(crag=CRAGConfig(enabled=True)),
            AdvancedGenerationConfig(flare=FLAREConfig(enabled=True)),
        ],
    )
    def test_unsupported_advanced_generation_returns_error_without_files(
        self,
        advanced: AdvancedGenerationConfig,
    ) -> None:
        result = generate(
            _cfg(generation=GenerationConfig(llm=OpenAILLMConfig(), advanced=advanced)),
            template_dir=STUB_TEMPLATE_DIR,
        )
        assert result.validation_passed is False
        assert result.files == {}
        assert any("generation.advanced" in error for error in result.errors)


# ─── 10. StrictUndefined ──────────────────────────────────────────────────────


class TestStrictUndefined:
    def test_undefined_variable_produces_error_result(self, tmp_path: Path) -> None:
        # Create a template that references an undefined variable
        stage_dir = tmp_path / "stages" / "chunking"
        stage_dir.mkdir(parents=True)
        (stage_dir / "recursive.py.j2").write_text(
            "# chunking\nUNDEFINED = {{ this_var_does_not_exist }}\n"
        )
        # Copy remaining needed templates from stub dir
        import shutil
        for sub in ["stages/embedding", "stages/vectordb", "stages/retrieval",
                    "stages/llm", "entrypoints"]:
            src = STUB_TEMPLATE_DIR / sub
            dst = tmp_path / sub
            if src.exists():
                shutil.copytree(src, dst)

        result = generate(_cfg(), template_dir=tmp_path)
        assert result.validation_passed is False
        assert len(result.errors) > 0


# ─── 11. Syntax error in generated Python ────────────────────────────────────


class TestSyntaxErrorInGeneratedPython:
    def test_bad_python_template_produces_error(self, tmp_path: Path) -> None:
        import shutil
        # Copy all stub templates
        shutil.copytree(STUB_TEMPLATE_DIR, tmp_path, dirs_exist_ok=True)
        # Overwrite pipeline.py.j2 with invalid Python
        bad_pipeline = tmp_path / "entrypoints" / "langchain" / "pipeline.py.j2"
        bad_pipeline.write_text("def broken_syntax(\n  # missing closing paren\n")

        result = generate(_cfg(framework=Framework.LANGCHAIN), template_dir=tmp_path)
        assert result.validation_passed is False
        assert any("pipeline.py" in e for e in result.errors)


# ─── 12. All optional fields None — no crash ─────────────────────────────────


class TestAllOptionalNone:
    def test_minimal_config_no_crash(self) -> None:
        cfg = RAGPipelineConfig(
            name="minimal",
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
            # pre_retrieval defaults to empty
            # post_retrieval defaults to no reranker
            # evaluation=None
            # generation.advanced=None
        )
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert isinstance(result, GeneratorResult)
        assert result.validation_passed is True

    def test_different_component_combinations(self) -> None:
        """Smoke-test several component combinations to verify no crashes."""
        combos = [
            # voyage + qdrant + hybrid_rrf
            _cfg(
                indexing=IndexingConfig(
                    embedding=VoyageEmbeddingConfig(),
                    vector_db=QdrantConfig(),
                ),
                retrieval=HybridRRFConfig(),
            ),
            # bge_m3 + milvus + dense
            _cfg(
                indexing=IndexingConfig(
                    embedding=BGEM3EmbeddingConfig(),
                    vector_db=MilvusConfig(),
                ),
                retrieval=DenseRetrievalConfig(),
            ),
            # ollama llm + chromadb
            _cfg(
                indexing=IndexingConfig(
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=ChromaDBConfig(),
                ),
                generation=GenerationConfig(llm=OllamaLLMConfig()),
            ),
            # late chunking + jina + qdrant
            _cfg(
                indexing=IndexingConfig(
                    chunking=LateChunkingConfig(),
                    embedding=JinaEmbeddingConfig(
                        model="jina-embeddings-v3",
                        late_chunking=True,
                    ),
                    vector_db=QdrantConfig(),
                ),
            ),
            # llamaindex + sentence_window
            _cfg(
                framework=Framework.LLAMAINDEX,
                retrieval=SentenceWindowConfig(),
            ),
            # anthropic llm
            _cfg(
                generation=GenerationConfig(llm=AnthropicLLMConfig()),
            ),
        ]
        for cfg in combos:
            result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
            assert result.validation_passed is True, (
                f"Failed for {cfg.name}: {result.errors}"
            )


# ─── 13. TemplateLoader unit tests ───────────────────────────────────────────


class TestTemplateLoader:
    def test_render_stage_valid(self) -> None:
        loader = TemplateLoader(STUB_TEMPLATE_DIR)
        from ragfactory.core.config import RecursiveChunkingConfig
        ctx = {
            "chunking": RecursiveChunkingConfig(),
            "config": _cfg(),
            "framework": "langchain",
            "pipeline_name": "test",
            "dependencies": [],
            "python_version": "3.11",
        }
        result = loader.render_stage("chunking", "recursive", ctx)
        assert "recursive" in result

    def test_render_stage_missing_template_raises_generator_error(self) -> None:
        loader = TemplateLoader(STUB_TEMPLATE_DIR)
        with pytest.raises(GeneratorError, match="Template not found"):
            loader.render_stage("chunking", "nonexistent_type", {})

    def test_render_entrypoint_missing_raises_generator_error(self) -> None:
        loader = TemplateLoader(STUB_TEMPLATE_DIR)
        with pytest.raises(GeneratorError, match="Template not found"):
            loader.render_entrypoint("langchain", "nonexistent", {})

    def test_render_common_missing_raises_generator_error(self) -> None:
        loader = TemplateLoader(STUB_TEMPLATE_DIR)
        with pytest.raises(GeneratorError, match="Template not found"):
            loader.render_common("nonexistent.j2", {})


# ─── 14. Stub template coverage meta-test (C2) ───────────────────────────────


import typing
from ragfactory.core import config as _cfg_mod


def _union_type_literals(union_annotation: object) -> list[str]:
    """Extract the `type` field default from every member of a discriminated union.

    Config unions are Annotated[Union[A, B, ...], FieldInfo(...)].
    get_args returns (Union[A, B, ...], FieldInfo); we need to peel one more
    layer to reach the concrete model classes.
    """
    # Peel Annotated wrapper: first arg is the Union
    outer = typing.get_args(union_annotation)
    if not outer:
        return []
    union = outer[0]  # Union[A, B, ...]
    result = []
    for member in typing.get_args(union):
        if hasattr(member, "model_fields") and "type" in member.model_fields:
            result.append(member.model_fields["type"].default)
    return result


_STUB_STAGE_UNIONS = [
    (_cfg_mod.ChunkingConfig,  "stages/chunking"),
    (_cfg_mod.EmbeddingConfig, "stages/embedding"),
    (_cfg_mod.VectorDBConfig,  "stages/vectordb"),
    (_cfg_mod.RetrievalConfig, "stages/retrieval"),
    (_cfg_mod.RerankerConfig,  "stages/reranker"),
    (_cfg_mod.LLMConfig,       "stages/llm"),
]

_stub_params = [
    (subdir, type_literal)
    for union, subdir in _STUB_STAGE_UNIONS
    for type_literal in _union_type_literals(union)
]


class TestStubTemplateCoverage:
    """Meta-test: every config type literal must have a corresponding stub .j2 file.

    Prevents the silent failure mode where a new component is added to config.py
    but no stub template is added, causing generator tests to silently fail
    (or worse: produce misleading GeneratorError messages in unrelated tests).
    """

    @pytest.mark.parametrize(
        "subdir,type_literal",
        _stub_params,
        ids=[f"{s}/{t}" for s, t in _stub_params],
    )
    def test_stub_exists(self, subdir: str, type_literal: str) -> None:
        stub = STUB_TEMPLATE_DIR / subdir / f"{type_literal}.py.j2"
        assert stub.exists(), (
            f"Missing stub template: {stub}\n"
            f"Add a minimal stub at:\n"
            f"  tests/fixtures/stub_templates/{subdir}/{type_literal}.py.j2"
        )


class TestEvaluationGeneration:
    def test_eval_generation_when_configured(self) -> None:
        from ragfactory.core.config import EvaluationConfig
        cfg = _cfg(evaluation=EvaluationConfig(framework="ragas"))
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert result.validation_passed is True
        assert "eval.py" in result.files
        assert "ragas" in result.files["eval.py"]

    def test_no_eval_generation_when_none(self) -> None:
        cfg = _cfg(evaluation=None)
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert result.validation_passed is True
        assert "eval.py" not in result.files


class TestAgenticGeneration:
    def test_agentic_flow_generates_correct_template(self) -> None:
        cfg = _cfg(flow_type="agentic")
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert result.validation_passed is True
        assert "pipeline.py" in result.files
        assert "pipeline_agentic" in result.files["pipeline.py"]

    def test_agentic_flow_generates_correct_template_llamaindex(self) -> None:
        cfg = _cfg(framework="llamaindex", flow_type="agentic")
        result = generate(cfg, template_dir=STUB_TEMPLATE_DIR)
        assert result.validation_passed is True
        assert "pipeline.py" in result.files
        assert "pipeline_agentic" in result.files["pipeline.py"]

    def test_agentic_dependencies_langchain(self) -> None:
        from ragfactory.core.versions import get_dependencies
        cfg = _cfg(framework="langchain", flow_type="agentic")
        deps = get_dependencies(cfg)
        assert any(d.startswith("langgraph") for d in deps)

    def test_agentic_dependencies_llamaindex(self) -> None:
        from ragfactory.core.versions import get_dependencies
        cfg = _cfg(framework="llamaindex", flow_type="agentic")
        deps = get_dependencies(cfg)
        assert not any(d.startswith("langgraph") for d in deps)

