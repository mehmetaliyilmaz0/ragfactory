"""
Unit tests for ragfactory.core.config — Pydantic v2 model hierarchy.

Coverage strategy:
  1. Valid construction — defaults work, required fields enforced
  2. StrictModel — extra fields raise ValidationError immediately
  3. Discriminated unions — wrong type fields rejected, correct type accepted
  4. Field validators — all custom validation logic exercised
  5. YAML round-trip — serialize → deserialize → model equality
  6. YAML file round-trip — to_yaml_file / from_yaml_file
  7. Edge cases — boundary values, None optionals, enum coercion
"""

from __future__ import annotations

import os
import tempfile

import pytest
from pydantic import ValidationError

from ragfactory.core.config import (
    AgenticConfig,
    AdvancedGenerationConfig,
    AnthropicLLMConfig,
    BGEM3EmbeddingConfig,
    BreakpointType,
    ChromaDBConfig,
    CohereLLMConfig,
    CohereEmbeddingConfig,
    CohereRerankerConfig,
    ColBERTRerankerConfig,
    ContextualChunkingConfig,
    CrossEncoderRerankerConfig,
    CRAGConfig,
    DenseRetrievalConfig,
    DistanceMetric,
    EvalFramework,
    EvaluationConfig,
    FileSourceConfig,
    FixedChunkingConfig,
    FLAREConfig,
    FlashRankRerankerConfig,
    Framework,
    GeminiEmbeddingConfig,
    GenerationConfig,
    HybridRRFConfig,
    HybridWeightedConfig,
    IndexingConfig,
    IngestionConfig,
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
    RAGPipelineConfig,
    RecursiveChunkingConfig,
    RoutingConfig,
    RouteDefinition,
    S3SourceConfig,
    SemanticChunkingConfig,
    SentenceWindowConfig,
    SmallToBigConfig,
    URLSourceConfig,
    VoyageEmbeddingConfig,
    WeaviateConfig,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _minimal_config(**overrides: object) -> RAGPipelineConfig:
    """Minimal valid RAGPipelineConfig with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "name": "test-pipeline",
        "indexing": IndexingConfig(
            embedding=OpenAIEmbeddingConfig(),
            vector_db=ChromaDBConfig(),
        ),
        "generation": GenerationConfig(llm=OpenAILLMConfig()),
    }
    defaults.update(overrides)
    return RAGPipelineConfig(**defaults)  # type: ignore[arg-type]


# ─── 1. Valid construction ─────────────────────────────────────────────────────


class TestValidConstruction:
    def test_minimal_config_builds(self) -> None:
        cfg = _minimal_config()
        assert cfg.name == "test-pipeline"
        assert cfg.version == "1.0"
        assert cfg.framework == "langchain"

    def test_flow_type_defaults_to_linear(self) -> None:
        cfg = _minimal_config()
        assert cfg.flow_type == "linear"

    def test_flow_type_accepted_choices(self) -> None:
        for choice in ["linear", "router", "agentic"]:
            cfg = _minimal_config(flow_type=choice)
            assert cfg.flow_type == choice

    def test_flow_type_invalid_choice_raises(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_config(flow_type="invalid_flow")

    def test_framework_enum_coercion(self) -> None:
        cfg = _minimal_config(framework=Framework.LLAMAINDEX)
        assert cfg.framework == "llamaindex"

    def test_framework_string_accepted(self) -> None:
        cfg = _minimal_config(framework="llamaindex")
        assert cfg.framework == "llamaindex"

    def test_version_defaults_to_1_0(self) -> None:
        cfg = _minimal_config()
        assert cfg.version == "1.0"

    def test_retrieval_defaults_to_hybrid_rrf(self) -> None:
        cfg = _minimal_config()
        assert cfg.retrieval.type == "hybrid_rrf"  # type: ignore[union-attr]

    def test_evaluation_is_optional(self) -> None:
        cfg = _minimal_config()
        assert cfg.evaluation is None

    def test_evaluation_can_be_set(self) -> None:
        cfg = _minimal_config(evaluation=EvaluationConfig())
        assert cfg.evaluation is not None
        assert cfg.evaluation.framework == "ragas"

    def test_pre_retrieval_defaults_to_empty(self) -> None:
        cfg = _minimal_config()
        assert cfg.pre_retrieval.query_rewriting is None
        assert cfg.pre_retrieval.hyde is None
        assert cfg.pre_retrieval.routing is None


# ─── 2. Name validation ───────────────────────────────────────────────────────


class TestNameValidation:
    @pytest.mark.parametrize("valid_name", [
        "my-pipeline",
        "rag1",
        "production-rag-v2",
        "a",
        "a1",
    ])
    def test_valid_names_accepted(self, valid_name: str) -> None:
        cfg = _minimal_config(name=valid_name)
        assert cfg.name == valid_name

    @pytest.mark.parametrize("invalid_name", [
        "My-Pipeline",      # uppercase
        "-starts-with-dash",
        "has spaces",
        "has_underscore",
        "has.dot",
        "",                 # empty
        "A" * 65,           # too long
    ])
    def test_invalid_names_rejected(self, invalid_name: str) -> None:
        with pytest.raises(ValidationError):
            _minimal_config(name=invalid_name)


# ─── 3. StrictModel extra fields ──────────────────────────────────────────────


class TestStrictModel:
    def test_extra_fields_on_root_config_raise(self) -> None:
        with pytest.raises(ValidationError, match="extra_fields_not_permitted|Extra inputs"):
            RAGPipelineConfig(
                name="test",
                indexing=IndexingConfig(
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=ChromaDBConfig(),
                ),
                generation=GenerationConfig(llm=OpenAILLMConfig()),
                unknown_field="bad",
            )

    def test_extra_fields_on_chunking_raise(self) -> None:
        with pytest.raises(ValidationError):
            FixedChunkingConfig(chunk_size=512, chunk_overlap=50, unknown="bad")

    def test_extra_fields_on_embedding_raise(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIEmbeddingConfig(model="text-embedding-3-small", secret_field="bad")


# ─── 4. Discriminated Unions ──────────────────────────────────────────────────


class TestDiscriminatedUnions:
    """Wrong sub-type fields must be rejected — no cross-contamination."""

    def test_hybrid_rrf_has_rrf_k(self) -> None:
        cfg = HybridRRFConfig(top_k=50, rrf_k=60)
        assert cfg.rrf_k == 60

    def test_dense_config_rejects_rrf_k(self) -> None:
        with pytest.raises(ValidationError):
            DenseRetrievalConfig(type="dense", top_k=20, rrf_k=60)  # type: ignore[call-arg]

    def test_hybrid_weighted_has_alpha(self) -> None:
        cfg = HybridWeightedConfig(alpha=0.7)
        assert cfg.alpha == 0.7

    def test_dense_config_rejects_alpha(self) -> None:
        with pytest.raises(ValidationError):
            DenseRetrievalConfig(type="dense", top_k=20, alpha=0.5)  # type: ignore[call-arg]

    def test_retrieval_discriminator_dense(self) -> None:
        cfg = _minimal_config(retrieval={"type": "dense", "top_k": 10})
        assert cfg.retrieval.type == "dense"  # type: ignore[union-attr]
        assert cfg.retrieval.top_k == 10  # type: ignore[union-attr]

    def test_retrieval_discriminator_hybrid_rrf(self) -> None:
        cfg = _minimal_config(retrieval={"type": "hybrid_rrf", "rrf_k": 40})
        assert cfg.retrieval.type == "hybrid_rrf"  # type: ignore[union-attr]

    def test_retrieval_discriminator_rejects_unknown_type(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_config(retrieval={"type": "nonexistent"})

    def test_embedding_discriminator_openai(self) -> None:
        idx = IndexingConfig(
            embedding={"type": "openai", "model": "text-embedding-3-large"},  # type: ignore[arg-type]
            vector_db=ChromaDBConfig(),
        )
        assert idx.embedding.type == "openai"

    def test_embedding_discriminator_voyage(self) -> None:
        idx = IndexingConfig(
            embedding={"type": "voyage"},  # type: ignore[arg-type]
            vector_db=QdrantConfig(),
        )
        assert idx.embedding.type == "voyage"

    def test_reranker_discriminator_cohere(self) -> None:
        post = PostRetrievalConfig(reranker={"type": "cohere"})  # type: ignore[arg-type]
        assert post.reranker is not None
        assert post.reranker.type == "cohere"


# ─── 5. Chunking validators ───────────────────────────────────────────────────


class TestChunkingValidators:
    def test_fixed_overlap_less_than_size(self) -> None:
        cfg = FixedChunkingConfig(chunk_size=512, chunk_overlap=100)
        assert cfg.chunk_overlap < cfg.chunk_size

    def test_fixed_overlap_equal_to_size_raises(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            FixedChunkingConfig(chunk_size=256, chunk_overlap=256)

    def test_fixed_overlap_greater_than_size_raises(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            FixedChunkingConfig(chunk_size=256, chunk_overlap=300)

    def test_recursive_overlap_validation(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            RecursiveChunkingConfig(chunk_size=512, chunk_overlap=512)

    def test_contextual_overlap_validation(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            ContextualChunkingConfig(chunk_size=800, chunk_overlap=900)

    def test_contextual_prompt_must_have_whole_document(self) -> None:
        with pytest.raises(ValidationError, match="WHOLE_DOCUMENT"):
            ContextualChunkingConfig(
                context_prompt="Summarize: {{CHUNK_CONTENT}}"
            )

    def test_contextual_prompt_must_have_chunk_content(self) -> None:
        with pytest.raises(ValidationError, match="CHUNK_CONTENT"):
            ContextualChunkingConfig(
                context_prompt="Document: {{WHOLE_DOCUMENT}}"
            )

    def test_contextual_default_prompt_is_valid(self) -> None:
        cfg = ContextualChunkingConfig()
        assert "{{WHOLE_DOCUMENT}}" in cfg.context_prompt
        assert "{{CHUNK_CONTENT}}" in cfg.context_prompt

    def test_semantic_valid(self) -> None:
        cfg = SemanticChunkingConfig(
            breakpoint_threshold_type=BreakpointType.PERCENTILE,
            breakpoint_threshold_amount=90.0,
        )
        assert cfg.breakpoint_threshold_amount == 90.0

    def test_late_chunking_valid(self) -> None:
        cfg = LateChunkingConfig(chunk_size=256)
        assert cfg.type == "late"

    def test_page_level_has_no_params(self) -> None:
        cfg = PageLevelChunkingConfig()
        assert cfg.type == "page_level"

    def test_proposition_valid(self) -> None:
        cfg = PropositionChunkingConfig(extraction_model="gpt-4o")
        assert cfg.extraction_model == "gpt-4o"


# ─── 6. Embedding validators ──────────────────────────────────────────────────


class TestEmbeddingValidators:
    def test_nomic_valid_dimensionality(self) -> None:
        for dim in [64, 128, 256, 512, 768]:
            cfg = NomicEmbeddingConfig(dimensionality=dim)
            assert cfg.dimensionality == dim

    def test_nomic_invalid_dimensionality_raises(self) -> None:
        with pytest.raises(ValidationError, match="dimensionality"):
            NomicEmbeddingConfig(dimensionality=100)

    def test_nomic_invalid_dimensionality_1000_raises(self) -> None:
        with pytest.raises(ValidationError, match="dimensionality"):
            NomicEmbeddingConfig(dimensionality=1000)

    def test_bge_m3_defaults(self) -> None:
        cfg = BGEM3EmbeddingConfig()
        assert cfg.use_fp16 is True
        assert cfg.batch_size == 32

    def test_jina_late_chunking_flag(self) -> None:
        cfg = JinaEmbeddingConfig(late_chunking=True)
        assert cfg.late_chunking is True

    def test_openai_dimensions_optional(self) -> None:
        cfg = OpenAIEmbeddingConfig()
        assert cfg.dimensions is None

    def test_cohere_input_type_default(self) -> None:
        cfg = CohereEmbeddingConfig()
        assert cfg.input_type == "search_document"


# ─── 7. Retrieval validators ──────────────────────────────────────────────────


class TestRetrievalValidators:
    def test_small_to_big_parent_larger_than_child(self) -> None:
        cfg = SmallToBigConfig(child_chunk_size=256, parent_chunk_size=1024)
        assert cfg.parent_chunk_size > cfg.child_chunk_size

    def test_small_to_big_parent_equal_to_child_raises(self) -> None:
        with pytest.raises(ValidationError, match="parent_chunk_size"):
            SmallToBigConfig(child_chunk_size=512, parent_chunk_size=512)

    def test_small_to_big_parent_smaller_than_child_raises(self) -> None:
        with pytest.raises(ValidationError, match="parent_chunk_size"):
            SmallToBigConfig(child_chunk_size=1024, parent_chunk_size=256)

    def test_sentence_window_defaults(self) -> None:
        cfg = SentenceWindowConfig()
        assert cfg.window_size == 5
        assert cfg.top_k == 5

    def test_hybrid_rrf_defaults(self) -> None:
        cfg = HybridRRFConfig()
        assert cfg.rrf_k == 60
        assert cfg.bm25_k1 == 1.2
        assert cfg.bm25_b == 0.75


# ─── 8. Generation validators ─────────────────────────────────────────────────


class TestGenerationValidators:
    def test_prompt_template_requires_context(self) -> None:
        with pytest.raises(ValidationError, match="context"):
            GenerationConfig(
                llm=OpenAILLMConfig(),
                prompt_template="Answer: {question}",  # missing {context}
            )

    def test_prompt_template_requires_question(self) -> None:
        with pytest.raises(ValidationError, match="question"):
            GenerationConfig(
                llm=OpenAILLMConfig(),
                prompt_template="Context: {context}",  # missing {question}
            )

    def test_prompt_template_valid(self) -> None:
        cfg = GenerationConfig(
            llm=OpenAILLMConfig(),
            prompt_template="Context: {context}\nQ: {question}\nA:",
        )
        assert "{context}" in cfg.prompt_template
        assert "{question}" in cfg.prompt_template

    def test_default_prompt_is_valid(self) -> None:
        cfg = GenerationConfig(llm=OpenAILLMConfig())
        assert "{context}" in cfg.prompt_template
        assert "{question}" in cfg.prompt_template

    def test_anthropic_temperature_max_is_1(self) -> None:
        with pytest.raises(ValidationError):
            AnthropicLLMConfig(temperature=1.5)  # Anthropic max is 1.0

    def test_openai_temperature_max_is_2(self) -> None:
        cfg = OpenAILLMConfig(temperature=1.9)
        assert cfg.temperature == 1.9

    def test_ollama_no_api_key(self) -> None:
        cfg = OllamaLLMConfig(model="llama3.2")
        assert cfg.type == "ollama"


# ─── 9. CRAG / FLARE / Agentic ────────────────────────────────────────────────


class TestAdvancedGeneration:
    def test_crag_defaults(self) -> None:
        cfg = CRAGConfig()
        assert cfg.enabled is True
        assert cfg.web_search_provider == "tavily"
        assert cfg.upper_threshold == 0.7

    def test_flare_defaults(self) -> None:
        cfg = FLAREConfig()
        assert cfg.enabled is True
        assert cfg.confidence_threshold == 0.5
        assert cfg.max_iterations == 5

    def test_agentic_defaults(self) -> None:
        cfg = AgenticConfig()
        assert cfg.enabled is True
        assert cfg.reflection_enabled is True
        assert cfg.tools == []

    def test_advanced_generation_all_none(self) -> None:
        cfg = AdvancedGenerationConfig()
        assert cfg.crag is None
        assert cfg.flare is None
        assert cfg.agentic is None

    def test_full_config_with_crag(self) -> None:
        cfg = _minimal_config(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(web_search_provider="serper")
                ),
            )
        )
        assert cfg.generation.advanced is not None
        assert cfg.generation.advanced.crag is not None
        assert cfg.generation.advanced.crag.web_search_provider == "serper"


# ─── 10. Vector DB configs ────────────────────────────────────────────────────


class TestVectorDBConfigs:
    def test_chromadb_defaults(self) -> None:
        cfg = ChromaDBConfig()
        assert cfg.type == "chromadb"
        assert cfg.persist_directory == ".chroma"

    def test_qdrant_defaults(self) -> None:
        cfg = QdrantConfig()
        assert cfg.url == "http://localhost:6333"
        assert cfg.distance_metric == "cosine"

    def test_pinecone_defaults(self) -> None:
        cfg = PineconeConfig()
        assert cfg.index_name == "ragfactory"

    def test_weaviate_class_name_pascal_case(self) -> None:
        cfg = WeaviateConfig(class_name="MyDocuments")
        assert cfg.class_name == "MyDocuments"

    def test_milvus_uri(self) -> None:
        cfg = MilvusConfig(uri="https://cloud.zilliz.com/my-cluster")
        assert "zilliz" in cfg.uri

    def test_pgvector_defaults(self) -> None:
        cfg = PgVectorConfig()
        assert cfg.type == "pgvector"

    def test_distance_metric_enum(self) -> None:
        for metric in [DistanceMetric.COSINE, DistanceMetric.DOT, DistanceMetric.EUCLIDEAN]:
            cfg = ChromaDBConfig(distance_metric=metric)
            assert cfg.distance_metric == metric.value


# ─── 11. Ingestion configs ────────────────────────────────────────────────────


class TestIngestionConfigs:
    def test_file_source_defaults(self) -> None:
        src = FileSourceConfig(path="/data/docs")
        assert src.glob == "**/*"
        assert src.recursive is True

    def test_s3_source_defaults(self) -> None:
        src = S3SourceConfig(bucket="my-bucket")
        assert src.region == "us-east-1"
        assert src.prefix == ""

    def test_url_source_requires_urls(self) -> None:
        with pytest.raises(ValidationError):
            URLSourceConfig(urls=[])

    def test_url_source_valid(self) -> None:
        src = URLSourceConfig(urls=["https://example.com/doc.pdf"])
        assert len(src.urls) == 1

    def test_ingestion_defaults_to_empty(self) -> None:
        ing = IngestionConfig()
        assert ing.sources == []
        assert ing.parser == "default"


# ─── 12. Routing config ───────────────────────────────────────────────────────


class TestRoutingConfig:
    def test_routing_requires_at_least_two_routes(self) -> None:
        with pytest.raises(ValidationError):
            RoutingConfig(routes=[RouteDefinition(name="r1", description="only one")])

    def test_routing_valid_with_two_routes(self) -> None:
        cfg = RoutingConfig(
            routes=[
                RouteDefinition(name="technical", description="Technical documentation queries"),
                RouteDefinition(name="general", description="General knowledge queries"),
            ]
        )
        assert len(cfg.routes) == 2

    def test_routing_confidence_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            RoutingConfig(
                routes=[
                    RouteDefinition(name="a", description="a"),
                    RouteDefinition(name="b", description="b"),
                ],
                confidence_threshold=1.5,  # out of range
            )


# ─── 13. Evaluation config ────────────────────────────────────────────────────


class TestEvaluationConfig:
    def test_eval_defaults(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.framework == "ragas"
        assert "faithfulness" in cfg.metrics
        assert cfg.num_test_cases == 50
        assert cfg.pass_threshold == 0.7

    def test_eval_framework_both(self) -> None:
        cfg = EvaluationConfig(framework=EvalFramework.BOTH)
        assert cfg.framework == "both"

    def test_eval_metrics_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=[])


# ─── 14. YAML Round-Trip ──────────────────────────────────────────────────────


class TestYAMLRoundTrip:
    def _full_config(self) -> RAGPipelineConfig:
        return RAGPipelineConfig(
            name="round-trip-test",
            framework=Framework.LANGCHAIN,
            ingestion=IngestionConfig(
                sources=[FileSourceConfig(path="/data")],
            ),
            indexing=IndexingConfig(
                chunking=RecursiveChunkingConfig(chunk_size=512, chunk_overlap=50),
                embedding=VoyageEmbeddingConfig(model="voyage-3-large"),
                vector_db=QdrantConfig(collection_name="test-col"),
            ),
            pre_retrieval=PreRetrievalConfig(),
            retrieval=HybridRRFConfig(top_k=100, rrf_k=60),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(top_n=5),
            ),
            generation=GenerationConfig(
                llm=AnthropicLLMConfig(model="claude-sonnet-4-6"),
            ),
            evaluation=EvaluationConfig(
                framework=EvalFramework.RAGAS,
                num_test_cases=100,
            ),
        )

    def test_to_yaml_returns_string(self) -> None:
        cfg = self._full_config()
        yaml_str = cfg.to_yaml()
        assert isinstance(yaml_str, str)
        assert "round-trip-test" in yaml_str

    def test_from_yaml_returns_equal_config(self) -> None:
        cfg = self._full_config()
        yaml_str = cfg.to_yaml()
        restored = RAGPipelineConfig.from_yaml(yaml_str)
        assert restored == cfg

    def test_round_trip_preserves_retrieval_type(self) -> None:
        cfg = self._full_config()
        restored = RAGPipelineConfig.from_yaml(cfg.to_yaml())
        assert restored.retrieval.type == "hybrid_rrf"  # type: ignore[union-attr]

    def test_round_trip_preserves_reranker_type(self) -> None:
        cfg = self._full_config()
        restored = RAGPipelineConfig.from_yaml(cfg.to_yaml())
        assert restored.post_retrieval.reranker is not None
        assert restored.post_retrieval.reranker.type == "cohere"

    def test_round_trip_with_dense_retrieval(self) -> None:
        cfg = _minimal_config(retrieval=DenseRetrievalConfig(top_k=15))
        restored = RAGPipelineConfig.from_yaml(cfg.to_yaml())
        assert restored.retrieval.type == "dense"  # type: ignore[union-attr]
        assert restored.retrieval.top_k == 15  # type: ignore[union-attr]

    def test_round_trip_with_sentence_window(self) -> None:
        cfg = _minimal_config(retrieval=SentenceWindowConfig(window_size=3, top_k=7))
        restored = RAGPipelineConfig.from_yaml(cfg.to_yaml())
        assert restored.retrieval.type == "sentence_window"  # type: ignore[union-attr]
        assert restored.retrieval.window_size == 3  # type: ignore[union-attr]

    def test_yaml_contains_no_secrets(self) -> None:
        """YAML output must never contain API keys — those come from env vars."""
        cfg = self._full_config()
        yaml_str = cfg.to_yaml()
        # These strings should never appear in config YAML
        for secret_key in ["sk-", "Bearer ", "password", "secret_key"]:
            assert secret_key not in yaml_str

    def test_yaml_version_field_present(self) -> None:
        cfg = _minimal_config()
        yaml_str = cfg.to_yaml()
        assert "version: '1.0'" in yaml_str or "version: \"1.0\"" in yaml_str or "version: 1.0" in yaml_str


# ─── 15. YAML File Round-Trip ─────────────────────────────────────────────────


class TestYAMLFileRoundTrip:
    def test_to_yaml_file_and_from_yaml_file(self) -> None:
        cfg = _minimal_config()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            cfg.to_yaml_file(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            restored = RAGPipelineConfig.from_yaml_file(tmp_path)
            assert restored == cfg
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            RAGPipelineConfig.from_yaml_file("/nonexistent/path/config.yaml")


# ─── 16. Full pipeline combinations (smoke tests) ─────────────────────────────


class TestFullPipelineCombinations:
    """Smoke-test the 4 approved MVP combinations from the architecture plan."""

    def test_quick_start_combination(self) -> None:
        """chromadb + openai-small + fixed chunking + dense retrieval"""
        cfg = RAGPipelineConfig(
            name="quick-start",
            indexing=IndexingConfig(
                chunking=FixedChunkingConfig(chunk_size=512, chunk_overlap=50),
                embedding=OpenAIEmbeddingConfig(model="text-embedding-3-small"),
                vector_db=ChromaDBConfig(),
            ),
            retrieval=DenseRetrievalConfig(top_k=5),
            generation=GenerationConfig(llm=OpenAILLMConfig(model="gpt-4o-mini")),
        )
        assert cfg.name == "quick-start"

    def test_production_standard_combination(self) -> None:
        """qdrant + voyage-3-large + recursive + hybrid-rrf + cohere-reranker"""
        cfg = RAGPipelineConfig(
            name="production-standard",
            indexing=IndexingConfig(
                chunking=RecursiveChunkingConfig(chunk_size=512, chunk_overlap=50),
                embedding=VoyageEmbeddingConfig(model="voyage-3-large"),
                vector_db=QdrantConfig(),
            ),
            retrieval=HybridRRFConfig(top_k=100, rrf_k=60),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(model="rerank-v3.5", top_n=5),
            ),
            generation=GenerationConfig(llm=AnthropicLLMConfig(model="claude-sonnet-4-6")),
        )
        yaml_str = cfg.to_yaml()
        restored = RAGPipelineConfig.from_yaml(yaml_str)
        assert restored == cfg

    def test_high_accuracy_combination(self) -> None:
        """qdrant + contextual chunking + voyage + hybrid-rrf + colbert"""
        cfg = RAGPipelineConfig(
            name="high-accuracy",
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(),
                embedding=VoyageEmbeddingConfig(model="voyage-3-large"),
                vector_db=QdrantConfig(collection_name="high-acc"),
            ),
            retrieval=HybridRRFConfig(),
            post_retrieval=PostRetrievalConfig(
                reranker=ColBERTRerankerConfig(top_n=5),
            ),
            generation=GenerationConfig(
                llm=AnthropicLLMConfig(model="claude-opus-4-6"),
            ),
        )
        assert cfg.indexing.chunking.type == "contextual"

    def test_local_dev_combination(self) -> None:
        """chromadb + bge-m3 + recursive + hybrid-rrf + flashrank + ollama"""
        cfg = RAGPipelineConfig(
            name="local-dev",
            indexing=IndexingConfig(
                chunking=RecursiveChunkingConfig(),
                embedding=BGEM3EmbeddingConfig(use_fp16=True),
                vector_db=ChromaDBConfig(),
            ),
            retrieval=HybridRRFConfig(),
            post_retrieval=PostRetrievalConfig(
                reranker=FlashRankRerankerConfig(top_n=5),
            ),
            generation=GenerationConfig(llm=OllamaLLMConfig(model="llama3.2")),
        )
        assert cfg.generation.llm.type == "ollama"

    def test_llamaindex_framework_accepted(self) -> None:
        cfg = RAGPipelineConfig(
            name="llamaindex-pipeline",
            framework=Framework.LLAMAINDEX,
            indexing=IndexingConfig(
                embedding=GeminiEmbeddingConfig(),
                vector_db=PineconeConfig(index_name="gemini-idx"),
            ),
            retrieval=SentenceWindowConfig(window_size=5),
            generation=GenerationConfig(llm=CohereLLMConfig(model="command-r-plus")),
        )
        assert cfg.framework == "llamaindex"
        yaml_str = cfg.to_yaml()
        restored = RAGPipelineConfig.from_yaml(yaml_str)
        assert restored.framework == "llamaindex"
