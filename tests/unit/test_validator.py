"""
Unit tests for ragfactory.core.validator.

Coverage:
  1. Hard incompatibility ERRORs (from INCOMPATIBLE list)
  2. Cross-field checks (late chunking flags, contextual chunking model)
  3. FLARE × LLM logprob edge cases
  4. Multiple advanced techniques warning
  5. Reranker top_n vs retrieval top_k warning
  6. WARNINGS passthrough (soft warnings → issues)
  7. Cost estimation (with and without corpus_tokens)
  8. ValidationResult structural properties
  9. Type safety
"""

from __future__ import annotations

import pytest

from ragfactory.core.config import (
    AdvancedGenerationConfig,
    AgenticConfig,
    AnthropicLLMConfig,
    BGEM3EmbeddingConfig,
    ChromaDBConfig,
    CohereLLMConfig,
    CohereEmbeddingConfig,
    CohereRerankerConfig,
    ColBERTRerankerConfig,
    ContextualChunkingConfig,
    CRAGConfig,
    CrossEncoderRerankerConfig,
    DenseRetrievalConfig,
    EvaluationConfig,
    FixedChunkingConfig,
    FLAREConfig,
    FlashRankRerankerConfig,
    Framework,
    GeminiEmbeddingConfig,
    GenerationConfig,
    HybridRRFConfig,
    HybridWeightedConfig,
    HyDEConfig,
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
    ParserType,
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
    SemanticChunkingConfig,
    SentenceWindowConfig,
    SmallToBigConfig,
    VoyageEmbeddingConfig,
    WeaviateConfig,
)
from ragfactory.core.validator import (
    CostEstimate,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    _is_active,
    validate,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _cfg(**overrides: object) -> RAGPipelineConfig:
    """Minimal valid config with sensible defaults."""
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


def _has_error(result: ValidationResult, code: str) -> bool:
    return any(i.code == code and i.severity == ValidationSeverity.ERROR for i in result.issues)


def _has_warning(result: ValidationResult, code: str) -> bool:
    return any(i.code == code and i.severity == ValidationSeverity.WARNING for i in result.issues)


def _has_info(result: ValidationResult, code: str) -> bool:
    return any(i.code == code and i.severity == ValidationSeverity.INFO for i in result.issues)


def _issue_codes(result: ValidationResult) -> set[str]:
    return {i.code for i in result.issues}


# ─── 1. ValidationResult structure ───────────────────────────────────────────


class TestValidationResultStructure:
    def test_valid_config_returns_valid_true(self) -> None:
        result = validate(_cfg())
        assert result.valid is True

    def test_valid_config_has_no_errors(self) -> None:
        result = validate(_cfg())
        assert result.errors == []

    def test_errors_property_filters_correctly(self) -> None:
        result = validate(_cfg(
            retrieval=HybridRRFConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),  # incompatible with hybrid
            ),
        ))
        assert all(i.severity == ValidationSeverity.ERROR for i in result.errors)

    def test_warnings_property_filters_correctly(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                embedding=BGEM3EmbeddingConfig(),  # triggers GPU warning
                vector_db=QdrantConfig(),
            ),
        ))
        assert all(i.severity == ValidationSeverity.WARNING for i in result.warnings)

    def test_error_config_returns_valid_false(self) -> None:
        result = validate(_cfg(
            retrieval=HybridRRFConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
        ))
        assert result.valid is False

    def test_wrong_input_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="RAGPipelineConfig"):
            validate("not a config")  # type: ignore[arg-type]

    def test_warnings_do_not_affect_valid(self) -> None:
        # BGE-M3 triggers a GPU warning but the pipeline is otherwise valid
        result = validate(_cfg(
            indexing=IndexingConfig(
                embedding=BGEM3EmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert result.valid is True
        assert len(result.warnings) > 0


# ─── 2. FLARE × LLM incompatibilities ────────────────────────────────────────


def _flare_config(llm: object) -> RAGPipelineConfig:
    return _cfg(
        generation=GenerationConfig(
            llm=llm,  # type: ignore[arg-type]
            advanced=AdvancedGenerationConfig(flare=FLAREConfig(enabled=True)),
        ),
    )


class TestFLAREIncompatibilities:
    def test_flare_anthropic_is_error(self) -> None:
        result = validate(_flare_config(AnthropicLLMConfig()))
        assert result.valid is False
        # Should have an error code containing ANTHROPIC
        assert any("ANTHROPIC" in i.code for i in result.errors)

    def test_flare_openai_is_valid(self) -> None:
        result = validate(_flare_config(OpenAILLMConfig()))
        # No FLARE-related errors (OpenAI supports logprobs)
        assert not any("FLARE" in i.code for i in result.errors)

    def test_flare_cohere_is_error(self) -> None:
        result = validate(_flare_config(CohereLLMConfig()))
        assert result.valid is False
        assert any("COHERE" in i.code for i in result.errors)

    def test_flare_ollama_is_warning_not_error(self) -> None:
        result = validate(_flare_config(OllamaLLMConfig(model="llama3.2")))
        # Should NOT be a hard error
        assert not any(
            "OLLAMA" in i.code and i.severity == ValidationSeverity.ERROR
            for i in result.issues
        )
        # Should be a WARNING
        assert _has_warning(result, "FLARE_OLLAMA_LOGPROBS_UNRELIABLE")

    def test_flare_disabled_no_issues(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=AnthropicLLMConfig(),
                advanced=AdvancedGenerationConfig(flare=FLAREConfig(enabled=False)),
            ),
        ))
        # FLARE disabled → no logprob error
        assert not any("FLARE" in i.code for i in result.issues)

    def test_flare_none_no_issues(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=AnthropicLLMConfig(),
                advanced=AdvancedGenerationConfig(flare=None),
            ),
        ))
        assert not any("FLARE" in i.code for i in result.issues)


# ─── 3. Late chunking cross-field checks ─────────────────────────────────────


class TestLateChunkingChecks:
    def test_late_with_openai_embedding_is_error(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=LateChunkingConfig(),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert result.valid is False

    def test_late_with_jina_v3_flag_true_is_valid(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=LateChunkingConfig(),
                embedding=JinaEmbeddingConfig(
                    model="jina-embeddings-v3",
                    late_chunking=True,
                ),
                vector_db=QdrantConfig(),
            ),
        ))
        # No late-chunking-specific errors
        late_errors = [
            i for i in result.errors
            if "LATE" in i.code or "JINA" in i.code
        ]
        assert late_errors == [], f"Unexpected errors: {late_errors}"

    def test_late_with_jina_v2_is_error(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=LateChunkingConfig(),
                embedding=JinaEmbeddingConfig(
                    model="jina-embeddings-v2-base-en",
                    late_chunking=True,
                ),
                vector_db=QdrantConfig(),
            ),
        ))
        assert result.valid is False
        assert _has_error(result, "LATE_CHUNKING_JINA_V2")

    def test_late_with_jina_v3_flag_false_is_error(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=LateChunkingConfig(),
                embedding=JinaEmbeddingConfig(
                    model="jina-embeddings-v3",
                    late_chunking=False,
                ),
                vector_db=QdrantConfig(),
            ),
        ))
        assert result.valid is False
        assert _has_error(result, "LATE_CHUNKING_FLAG_NOT_SET")

    def test_late_with_voyage_embedding_is_error(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=LateChunkingConfig(),
                embedding=VoyageEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert result.valid is False


# ─── 4. Hybrid search × vector DB ────────────────────────────────────────────


class TestHybridSearchVectorDB:
    def test_hybrid_rrf_chromadb_is_error(self) -> None:
        result = validate(_cfg(
            retrieval=HybridRRFConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
        ))
        assert result.valid is False
        assert any("CHROMADB" in i.code for i in result.errors)

    def test_hybrid_rrf_qdrant_is_valid(self) -> None:
        result = validate(_cfg(
            retrieval=HybridRRFConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert not any("CHROMADB" in i.code or "PINECONE" in i.code for i in result.errors)

    def test_hybrid_weighted_pinecone_is_error(self) -> None:
        result = validate(_cfg(
            retrieval=HybridWeightedConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=PineconeConfig(),
            ),
        ))
        assert result.valid is False
        assert any("PINECONE" in i.code for i in result.errors)

    def test_hybrid_rrf_milvus_is_valid(self) -> None:
        result = validate(_cfg(
            retrieval=HybridRRFConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=MilvusConfig(),
            ),
        ))
        hybrid_errors = [i for i in result.errors if "CHROMADB" in i.code or "PINECONE" in i.code]
        assert hybrid_errors == []

    def test_dense_chromadb_is_valid(self) -> None:
        result = validate(_cfg(
            retrieval=DenseRetrievalConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
        ))
        # Dense + ChromaDB is fine — no hybrid errors
        hybrid_errors = [i for i in result.errors if "CHROMADB" in i.code]
        assert hybrid_errors == []


# ─── 5. Sentence window × framework ──────────────────────────────────────────


class TestSentenceWindowFramework:
    def test_sentence_window_langchain_is_error(self) -> None:
        result = validate(_cfg(
            framework=Framework.LANGCHAIN,
            retrieval=SentenceWindowConfig(),
        ))
        assert result.valid is False
        assert any("LANGCHAIN" in i.code for i in result.errors)

    def test_sentence_window_llamaindex_is_valid(self) -> None:
        result = validate(_cfg(
            framework=Framework.LLAMAINDEX,
            retrieval=SentenceWindowConfig(),
        ))
        sentence_errors = [i for i in result.errors if "SENTENCE" in i.code or "LANGCHAIN" in i.code]
        assert sentence_errors == []


# ─── 6. Multiple advanced techniques ─────────────────────────────────────────


class TestMultipleAdvancedTechniques:
    def test_crag_only_no_warning(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(crag=CRAGConfig(enabled=True)),
            ),
        ))
        assert not _has_warning(result, "MULTIPLE_ADVANCED_TECHNIQUES")

    def test_crag_and_flare_both_enabled_warns(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),  # OpenAI so FLARE doesn't add an ERROR too
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(enabled=True),
                    flare=FLAREConfig(enabled=True),
                ),
            ),
        ))
        assert _has_warning(result, "MULTIPLE_ADVANCED_TECHNIQUES")

    def test_all_three_advanced_warns(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(enabled=True),
                    flare=FLAREConfig(enabled=True),
                    agentic=AgenticConfig(enabled=True),
                ),
            ),
        ))
        assert _has_warning(result, "MULTIPLE_ADVANCED_TECHNIQUES")

    def test_all_disabled_no_warning(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(enabled=False),
                    flare=FLAREConfig(enabled=False),
                    agentic=AgenticConfig(enabled=False),
                ),
            ),
        ))
        assert not _has_warning(result, "MULTIPLE_ADVANCED_TECHNIQUES")

    def test_advanced_none_no_warning(self) -> None:
        result = validate(_cfg())
        assert not _has_warning(result, "MULTIPLE_ADVANCED_TECHNIQUES")


# ─── 7. Reranker top_n vs retrieval top_k ────────────────────────────────────


class TestRerankerTopNCheck:
    def test_top_n_equals_top_k_warns(self) -> None:
        result = validate(_cfg(
            retrieval=DenseRetrievalConfig(top_k=5),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(top_n=5),
            ),
        ))
        assert _has_warning(result, "RERANKER_TOP_N_EXCEEDS_TOP_K")

    def test_top_n_greater_than_top_k_warns(self) -> None:
        result = validate(_cfg(
            retrieval=DenseRetrievalConfig(top_k=5),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(top_n=10),
            ),
        ))
        assert _has_warning(result, "RERANKER_TOP_N_EXCEEDS_TOP_K")

    def test_top_n_less_than_top_k_no_warning(self) -> None:
        result = validate(_cfg(
            retrieval=DenseRetrievalConfig(top_k=20),
            post_retrieval=PostRetrievalConfig(
                reranker=CohereRerankerConfig(top_n=5),
            ),
        ))
        assert not _has_warning(result, "RERANKER_TOP_N_EXCEEDS_TOP_K")

    def test_no_reranker_no_warning(self) -> None:
        result = validate(_cfg(
            retrieval=DenseRetrievalConfig(top_k=5),
            post_retrieval=PostRetrievalConfig(reranker=None),
        ))
        assert not _has_warning(result, "RERANKER_TOP_N_EXCEEDS_TOP_K")

    def test_colbert_reranker_also_checked(self) -> None:
        result = validate(_cfg(
            retrieval=DenseRetrievalConfig(top_k=3),
            post_retrieval=PostRetrievalConfig(
                reranker=ColBERTRerankerConfig(top_n=5),
            ),
        ))
        assert _has_warning(result, "RERANKER_TOP_N_EXCEEDS_TOP_K")


# ─── 8. Contextual chunking cross-field ──────────────────────────────────────


class TestContextualChunkingChecks:
    def test_contextual_with_ollama_context_model_warns(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(context_model="llama3.2"),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")

    def test_contextual_with_claude_haiku_no_throughput_warning(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(context_model="claude-3-haiku-20240307"),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert not _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")

    def test_contextual_extra_api_key_info_when_providers_differ(self) -> None:
        # Main LLM is OpenAI, context model is Claude → extra Anthropic key needed
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(context_model="claude-3-haiku-20240307"),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            generation=GenerationConfig(llm=OpenAILLMConfig()),
        ))
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")

    def test_contextual_same_provider_no_extra_key_info(self) -> None:
        # Main LLM is Anthropic, context model is Claude → same provider
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(context_model="claude-3-haiku-20240307"),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        ))
        assert not _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")

    def test_contextual_gpt_context_model_with_anthropic_llm(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(context_model="gpt-4o-mini"),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            generation=GenerationConfig(llm=AnthropicLLMConfig()),
        ))
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")


# ─── 9. Soft warnings passthrough ────────────────────────────────────────────


class TestSoftWarningsPassthrough:
    def test_chromadb_warning_emitted(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
        ))
        assert any("CHROMADB" in i.code for i in result.issues)

    def test_bge_m3_gpu_warning_emitted(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                embedding=BGEM3EmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        assert any("BGE_M3" in i.code or "bge_m3".upper() in i.code.upper() for i in result.issues)

    def test_hyde_info_emitted(self) -> None:
        result = validate(_cfg(
            pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=True)),
        ))
        assert any("HYDE" in i.code for i in result.issues)

    def test_hyde_disabled_no_issue(self) -> None:
        result = validate(_cfg(
            pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=False)),
        ))
        assert not any("HYDE" in i.code for i in result.issues)

    def test_crag_web_search_warning(self) -> None:
        result = validate(_cfg(
            generation=GenerationConfig(
                llm=OpenAILLMConfig(),
                advanced=AdvancedGenerationConfig(
                    crag=CRAGConfig(enabled=True, web_search_fallback=True),
                ),
            ),
        ))
        assert any("WEB_SEARCH" in i.code or "CRAG" in i.code for i in result.issues)


# ─── 10. Cost estimation ──────────────────────────────────────────────────────


class TestCostEstimation:
    def test_simple_pipeline_no_costs(self) -> None:
        result = validate(_cfg())
        assert result.costs == []

    def test_contextual_chunking_cost_no_corpus(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        cost_items = [c for c in result.costs if c.component == "contextual_chunking"]
        assert len(cost_items) == 1
        c = cost_items[0]
        assert c.cost_per_million_tokens == pytest.approx(1.02)
        assert c.estimated_total is None

    def test_contextual_chunking_cost_with_corpus(self) -> None:
        result = validate(
            _cfg(
                indexing=IndexingConfig(
                    chunking=ContextualChunkingConfig(),
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=QdrantConfig(),
                ),
            ),
            corpus_tokens=1_000_000,
        )
        cost_items = [c for c in result.costs if c.component == "contextual_chunking"]
        assert len(cost_items) == 1
        assert cost_items[0].estimated_total == pytest.approx(1.02)

    def test_contextual_chunking_cost_large_corpus(self) -> None:
        result = validate(
            _cfg(
                indexing=IndexingConfig(
                    chunking=ContextualChunkingConfig(),
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=QdrantConfig(),
                ),
            ),
            corpus_tokens=10_000_000,
        )
        cost_items = [c for c in result.costs if c.component == "contextual_chunking"]
        assert cost_items[0].estimated_total == pytest.approx(10.20)

    def test_proposition_chunking_cost(self) -> None:
        result = validate(
            _cfg(
                indexing=IndexingConfig(
                    chunking=PropositionChunkingConfig(),
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=QdrantConfig(),
                ),
            ),
            corpus_tokens=1_000_000,
        )
        cost_items = [c for c in result.costs if c.component == "proposition_chunking"]
        assert len(cost_items) == 1
        assert cost_items[0].cost_per_million_tokens == pytest.approx(2.50)
        assert cost_items[0].estimated_total == pytest.approx(2.50)

    def test_corpus_tokens_zero(self) -> None:
        result = validate(
            _cfg(
                indexing=IndexingConfig(
                    chunking=ContextualChunkingConfig(),
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=QdrantConfig(),
                ),
            ),
            corpus_tokens=0,
        )
        cost_items = [c for c in result.costs if c.component == "contextual_chunking"]
        assert cost_items[0].estimated_total == pytest.approx(0.0)

    def test_cost_estimate_type(self) -> None:
        result = validate(
            _cfg(
                indexing=IndexingConfig(
                    chunking=ContextualChunkingConfig(),
                    embedding=OpenAIEmbeddingConfig(),
                    vector_db=QdrantConfig(),
                ),
            ),
        )
        for c in result.costs:
            assert isinstance(c, CostEstimate)


# ─── 11. Issue structure ──────────────────────────────────────────────────────


class TestIssueStructure:
    def test_error_has_required_fields(self) -> None:
        result = validate(_cfg(
            retrieval=HybridRRFConfig(),
            indexing=IndexingConfig(
                embedding=OpenAIEmbeddingConfig(),
                vector_db=ChromaDBConfig(),
            ),
        ))
        for issue in result.errors:
            assert issue.code
            assert issue.message
            assert issue.component_path
            assert issue.severity == ValidationSeverity.ERROR

    def test_warning_issue_type(self) -> None:
        result = validate(_cfg(
            indexing=IndexingConfig(
                embedding=BGEM3EmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
        ))
        for w in result.warnings:
            assert isinstance(w, ValidationIssue)
            assert w.severity == ValidationSeverity.WARNING


# ─── 12. _is_active direct unit tests (M1) ───────────────────────────────────


class TestIsActive:
    """Direct isolation tests for the _is_active dot-path resolver.

    Each path pattern in the function has at least one True and one False test
    so a regression in _is_active fails loudly here rather than silently
    missing a validator check downstream.
    """

    # ── framework ─────────────────────────────────────────────────────────────

    def test_framework_langchain_true(self) -> None:
        assert _is_active(_cfg(framework=Framework.LANGCHAIN), "framework.langchain") is True

    def test_framework_langchain_false_when_llamaindex(self) -> None:
        assert _is_active(_cfg(framework=Framework.LLAMAINDEX), "framework.langchain") is False

    def test_framework_llamaindex_true(self) -> None:
        assert _is_active(_cfg(framework=Framework.LLAMAINDEX), "framework.llamaindex") is True

    def test_framework_llamaindex_false_when_langchain(self) -> None:
        assert _is_active(_cfg(framework=Framework.LANGCHAIN), "framework.llamaindex") is False

    # ── indexing.chunking ─────────────────────────────────────────────────────

    def _chunking_cfg(self, chunking: object) -> RAGPipelineConfig:
        return _cfg(indexing=IndexingConfig(
            chunking=chunking,  # type: ignore[arg-type]
            embedding=OpenAIEmbeddingConfig(),
            vector_db=QdrantConfig(),
        ))

    def test_chunking_fixed(self) -> None:
        c = self._chunking_cfg(FixedChunkingConfig())
        assert _is_active(c, "indexing.chunking.fixed") is True
        assert _is_active(c, "indexing.chunking.recursive") is False

    def test_chunking_recursive(self) -> None:
        c = self._chunking_cfg(RecursiveChunkingConfig())
        assert _is_active(c, "indexing.chunking.recursive") is True
        assert _is_active(c, "indexing.chunking.fixed") is False

    def test_chunking_semantic(self) -> None:
        c = self._chunking_cfg(SemanticChunkingConfig())
        assert _is_active(c, "indexing.chunking.semantic") is True

    def test_chunking_contextual(self) -> None:
        c = self._chunking_cfg(ContextualChunkingConfig())
        assert _is_active(c, "indexing.chunking.contextual") is True

    def test_chunking_late(self) -> None:
        c = self._chunking_cfg(LateChunkingConfig())
        assert _is_active(c, "indexing.chunking.late") is True

    def test_chunking_page_level(self) -> None:
        c = self._chunking_cfg(PageLevelChunkingConfig())
        assert _is_active(c, "indexing.chunking.page_level") is True

    def test_chunking_proposition(self) -> None:
        c = self._chunking_cfg(PropositionChunkingConfig())
        assert _is_active(c, "indexing.chunking.proposition") is True

    # ── indexing.embedding ────────────────────────────────────────────────────

    def _emb_cfg(self, embedding: object) -> RAGPipelineConfig:
        return _cfg(indexing=IndexingConfig(
            embedding=embedding,  # type: ignore[arg-type]
            vector_db=QdrantConfig(),
        ))

    def test_embedding_openai(self) -> None:
        c = self._emb_cfg(OpenAIEmbeddingConfig())
        assert _is_active(c, "indexing.embedding.openai") is True
        assert _is_active(c, "indexing.embedding.jina") is False

    def test_embedding_cohere(self) -> None:
        assert _is_active(self._emb_cfg(CohereEmbeddingConfig()), "indexing.embedding.cohere") is True

    def test_embedding_voyage(self) -> None:
        assert _is_active(self._emb_cfg(VoyageEmbeddingConfig()), "indexing.embedding.voyage") is True

    def test_embedding_gemini(self) -> None:
        assert _is_active(self._emb_cfg(GeminiEmbeddingConfig()), "indexing.embedding.gemini") is True

    def test_embedding_bge_m3(self) -> None:
        assert _is_active(self._emb_cfg(BGEM3EmbeddingConfig()), "indexing.embedding.bge_m3") is True

    def test_embedding_nomic(self) -> None:
        assert _is_active(self._emb_cfg(NomicEmbeddingConfig()), "indexing.embedding.nomic") is True

    def test_embedding_jina(self) -> None:
        assert _is_active(self._emb_cfg(JinaEmbeddingConfig()), "indexing.embedding.jina") is True

    # ── indexing.vector_db ────────────────────────────────────────────────────

    def _vdb_cfg(self, vector_db: object) -> RAGPipelineConfig:
        return _cfg(indexing=IndexingConfig(
            embedding=OpenAIEmbeddingConfig(),
            vector_db=vector_db,  # type: ignore[arg-type]
        ))

    def test_vectordb_chromadb(self) -> None:
        c = self._vdb_cfg(ChromaDBConfig())
        assert _is_active(c, "indexing.vector_db.chromadb") is True
        assert _is_active(c, "indexing.vector_db.qdrant") is False

    def test_vectordb_qdrant(self) -> None:
        assert _is_active(self._vdb_cfg(QdrantConfig()), "indexing.vector_db.qdrant") is True

    def test_vectordb_pinecone(self) -> None:
        assert _is_active(self._vdb_cfg(PineconeConfig()), "indexing.vector_db.pinecone") is True

    def test_vectordb_weaviate(self) -> None:
        assert _is_active(self._vdb_cfg(WeaviateConfig()), "indexing.vector_db.weaviate") is True

    def test_vectordb_milvus(self) -> None:
        assert _is_active(self._vdb_cfg(MilvusConfig()), "indexing.vector_db.milvus") is True

    def test_vectordb_pgvector(self) -> None:
        assert _is_active(self._vdb_cfg(PgVectorConfig()), "indexing.vector_db.pgvector") is True

    # ── retrieval ─────────────────────────────────────────────────────────────

    def test_retrieval_dense(self) -> None:
        c = _cfg(retrieval=DenseRetrievalConfig())
        assert _is_active(c, "retrieval.dense") is True
        assert _is_active(c, "retrieval.hybrid_rrf") is False

    def test_retrieval_hybrid_rrf(self) -> None:
        assert _is_active(_cfg(retrieval=HybridRRFConfig()), "retrieval.hybrid_rrf") is True

    def test_retrieval_hybrid_weighted(self) -> None:
        assert _is_active(_cfg(retrieval=HybridWeightedConfig()), "retrieval.hybrid_weighted") is True

    def test_retrieval_small_to_big(self) -> None:
        assert _is_active(_cfg(retrieval=SmallToBigConfig()), "retrieval.small_to_big") is True

    def test_retrieval_sentence_window(self) -> None:
        assert _is_active(_cfg(retrieval=SentenceWindowConfig()), "retrieval.sentence_window") is True

    # ── pre_retrieval ─────────────────────────────────────────────────────────

    def test_pre_retrieval_hyde_enabled(self) -> None:
        c = _cfg(pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=True)))
        assert _is_active(c, "pre_retrieval.hyde") is True

    def test_pre_retrieval_hyde_disabled(self) -> None:
        c = _cfg(pre_retrieval=PreRetrievalConfig(hyde=HyDEConfig(enabled=False)))
        assert _is_active(c, "pre_retrieval.hyde") is False

    def test_pre_retrieval_hyde_none(self) -> None:
        c = _cfg(pre_retrieval=PreRetrievalConfig(hyde=None))
        assert _is_active(c, "pre_retrieval.hyde") is False

    def test_pre_retrieval_query_rewriting_enabled(self) -> None:
        from ragfactory.core.config import QueryRewritingConfig
        c = _cfg(pre_retrieval=PreRetrievalConfig(query_rewriting=QueryRewritingConfig(enabled=True)))
        assert _is_active(c, "pre_retrieval.query_rewriting") is True

    def test_pre_retrieval_query_rewriting_none(self) -> None:
        c = _cfg(pre_retrieval=PreRetrievalConfig(query_rewriting=None))
        assert _is_active(c, "pre_retrieval.query_rewriting") is False

    def test_pre_retrieval_routing_enabled(self) -> None:
        c = _cfg(pre_retrieval=PreRetrievalConfig(
            routing=RoutingConfig(
                enabled=True,
                routes=[
                    RouteDefinition(name="a", description="route a"),
                    RouteDefinition(name="b", description="route b"),
                ],
            )
        ))
        assert _is_active(c, "pre_retrieval.routing") is True

    def test_pre_retrieval_routing_none(self) -> None:
        c = _cfg(pre_retrieval=PreRetrievalConfig(routing=None))
        assert _is_active(c, "pre_retrieval.routing") is False

    # ── post_retrieval.reranker ───────────────────────────────────────────────

    def test_reranker_cohere(self) -> None:
        c = _cfg(post_retrieval=PostRetrievalConfig(reranker=CohereRerankerConfig()))
        assert _is_active(c, "post_retrieval.reranker.cohere") is True
        assert _is_active(c, "post_retrieval.reranker.colbert") is False

    def test_reranker_cross_encoder(self) -> None:
        c = _cfg(post_retrieval=PostRetrievalConfig(reranker=CrossEncoderRerankerConfig()))
        assert _is_active(c, "post_retrieval.reranker.cross_encoder") is True

    def test_reranker_colbert(self) -> None:
        c = _cfg(post_retrieval=PostRetrievalConfig(reranker=ColBERTRerankerConfig()))
        assert _is_active(c, "post_retrieval.reranker.colbert") is True

    def test_reranker_flashrank(self) -> None:
        c = _cfg(post_retrieval=PostRetrievalConfig(reranker=FlashRankRerankerConfig()))
        assert _is_active(c, "post_retrieval.reranker.flashrank") is True

    def test_reranker_none(self) -> None:
        c = _cfg(post_retrieval=PostRetrievalConfig(reranker=None))
        assert _is_active(c, "post_retrieval.reranker.cohere") is False

    # ── generation.llm ────────────────────────────────────────────────────────

    def test_llm_openai(self) -> None:
        c = _cfg(generation=GenerationConfig(llm=OpenAILLMConfig()))
        assert _is_active(c, "generation.llm.openai") is True
        assert _is_active(c, "generation.llm.anthropic") is False

    def test_llm_anthropic(self) -> None:
        c = _cfg(generation=GenerationConfig(llm=AnthropicLLMConfig()))
        assert _is_active(c, "generation.llm.anthropic") is True

    def test_llm_cohere_llm(self) -> None:
        c = _cfg(generation=GenerationConfig(llm=CohereLLMConfig()))
        assert _is_active(c, "generation.llm.cohere_llm") is True

    def test_llm_ollama(self) -> None:
        c = _cfg(generation=GenerationConfig(llm=OllamaLLMConfig()))
        assert _is_active(c, "generation.llm.ollama") is True

    # ── generation.advanced ───────────────────────────────────────────────────

    def test_advanced_flare_enabled(self) -> None:
        c = _cfg(generation=GenerationConfig(
            llm=OpenAILLMConfig(),
            advanced=AdvancedGenerationConfig(flare=FLAREConfig(enabled=True)),
        ))
        assert _is_active(c, "generation.advanced.flare") is True

    def test_advanced_flare_disabled(self) -> None:
        c = _cfg(generation=GenerationConfig(
            llm=OpenAILLMConfig(),
            advanced=AdvancedGenerationConfig(flare=FLAREConfig(enabled=False)),
        ))
        assert _is_active(c, "generation.advanced.flare") is False

    def test_advanced_flare_none_advanced(self) -> None:
        c = _cfg(generation=GenerationConfig(llm=OpenAILLMConfig(), advanced=None))
        assert _is_active(c, "generation.advanced.flare") is False

    def test_advanced_crag_enabled(self) -> None:
        c = _cfg(generation=GenerationConfig(
            llm=OpenAILLMConfig(),
            advanced=AdvancedGenerationConfig(crag=CRAGConfig(enabled=True)),
        ))
        assert _is_active(c, "generation.advanced.crag") is True

    def test_advanced_agentic_enabled(self) -> None:
        c = _cfg(generation=GenerationConfig(
            llm=OpenAILLMConfig(),
            advanced=AdvancedGenerationConfig(agentic=AgenticConfig(enabled=True)),
        ))
        assert _is_active(c, "generation.advanced.agentic") is True

    def test_advanced_crag_web_search_fallback_true(self) -> None:
        c = _cfg(generation=GenerationConfig(
            llm=OpenAILLMConfig(),
            advanced=AdvancedGenerationConfig(
                crag=CRAGConfig(enabled=True, web_search_fallback=True),
            ),
        ))
        assert _is_active(c, "generation.advanced.crag.web_search_fallback") is True

    def test_advanced_crag_web_search_fallback_false(self) -> None:
        c = _cfg(generation=GenerationConfig(
            llm=OpenAILLMConfig(),
            advanced=AdvancedGenerationConfig(
                crag=CRAGConfig(enabled=True, web_search_fallback=False),
            ),
        ))
        assert _is_active(c, "generation.advanced.crag.web_search_fallback") is False

    # ── evaluation ────────────────────────────────────────────────────────────

    def test_evaluation_present(self) -> None:
        c = _cfg(evaluation=EvaluationConfig())
        assert _is_active(c, "evaluation") is True

    def test_evaluation_none(self) -> None:
        c = _cfg(evaluation=None)
        assert _is_active(c, "evaluation") is False

    # ── ingestion.parser ──────────────────────────────────────────────────────

    def test_ingestion_parser_default(self) -> None:
        c = _cfg(ingestion=IngestionConfig(parser=ParserType.DEFAULT))
        assert _is_active(c, "ingestion.parser.default") is True
        assert _is_active(c, "ingestion.parser.unstructured") is False

    def test_ingestion_parser_unstructured(self) -> None:
        c = _cfg(ingestion=IngestionConfig(parser=ParserType.UNSTRUCTURED))
        assert _is_active(c, "ingestion.parser.unstructured") is True

    def test_ingestion_parser_azure_doc_intelligence(self) -> None:
        c = _cfg(ingestion=IngestionConfig(parser=ParserType.AZURE_DOC_INTELLIGENCE))
        assert _is_active(c, "ingestion.parser.azure_doc_intelligence") is True

    def test_ingestion_parser_docling(self) -> None:
        c = _cfg(ingestion=IngestionConfig(parser=ParserType.DOCLING))
        assert _is_active(c, "ingestion.parser.docling") is True

    # ── unknown path ──────────────────────────────────────────────────────────

    def test_unknown_path_returns_false(self) -> None:
        assert _is_active(_cfg(), "does.not.exist") is False

    def test_partial_known_prefix_returns_false(self) -> None:
        assert _is_active(_cfg(), "indexing.chunking") is False


# ─── 13. Contextual chunking cloud model regression (M3) ─────────────────────


class TestContextualChunkingCloudModels:
    """Regression tests for the M3 fix — cloud models must NOT trigger the
    local-model slow-inference warning, and MUST trigger the extra-API-key INFO
    when their provider differs from the main LLM.
    """

    def _ctx_cfg(self, context_model: str, llm: object = None) -> RAGPipelineConfig:
        return _cfg(
            indexing=IndexingConfig(
                chunking=ContextualChunkingConfig(context_model=context_model),
                embedding=OpenAIEmbeddingConfig(),
                vector_db=QdrantConfig(),
            ),
            generation=GenerationConfig(llm=llm or OpenAILLMConfig()),  # type: ignore[arg-type]
        )

    def test_gemini_does_not_trigger_local_warning(self) -> None:
        result = validate(self._ctx_cfg("gemini-1.5-flash"))
        assert not _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")

    def test_gemini_triggers_extra_api_key_info(self) -> None:
        # context=gemini, LLM=openai → different provider → needs GOOGLE_API_KEY
        result = validate(self._ctx_cfg("gemini-1.5-flash"))
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        info = next(i for i in result.issues if i.code == "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        assert "GOOGLE_API_KEY" in info.message

    def test_command_r_plus_triggers_extra_api_key_info(self) -> None:
        # context=cohere command-r-plus, LLM=openai → needs COHERE_API_KEY
        result = validate(self._ctx_cfg("command-r-plus"))
        assert not _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        info = next(i for i in result.issues if i.code == "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        assert "COHERE_API_KEY" in info.message

    def test_mistral_large_triggers_extra_api_key_info(self) -> None:
        result = validate(self._ctx_cfg("mistral-large"))
        assert not _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        info = next(i for i in result.issues if i.code == "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        assert "MISTRAL_API_KEY" in info.message

    def test_llama32_still_triggers_local_warning(self) -> None:
        # Regression: bare Ollama model names must still emit the slow-local warning
        result = validate(self._ctx_cfg("llama3.2"))
        assert _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")
        assert not _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")

    def test_gpt4o_mini_same_provider_no_issues(self) -> None:
        # context=gpt-4o-mini, LLM=openai → same provider → no extra key needed
        result = validate(self._ctx_cfg("gpt-4o-mini", llm=OpenAILLMConfig()))
        assert not _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")
        assert not _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")

    def test_claude_haiku_with_openai_llm_triggers_extra_key(self) -> None:
        result = validate(self._ctx_cfg("claude-3-haiku-20240307", llm=OpenAILLMConfig()))
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        info = next(i for i in result.issues if i.code == "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        assert "ANTHROPIC_API_KEY" in info.message

    def test_o1_model_recognised_as_openai(self) -> None:
        # o1 prefix is a known OpenAI model family
        result = validate(self._ctx_cfg("o1-preview", llm=AnthropicLLMConfig()))
        assert not _has_warning(result, "CONTEXTUAL_CHUNKING_SLOW_LOCAL_MODEL")
        assert _has_info(result, "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        info = next(i for i in result.issues if i.code == "CONTEXTUAL_CHUNKING_EXTRA_API_KEY")
        assert "OPENAI_API_KEY" in info.message
