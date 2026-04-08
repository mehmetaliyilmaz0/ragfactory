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
    CohereRerankerConfig,
    ColBERTRerankerConfig,
    ContextualChunkingConfig,
    CRAGConfig,
    DenseRetrievalConfig,
    FLAREConfig,
    Framework,
    GenerationConfig,
    HybridRRFConfig,
    HybridWeightedConfig,
    HyDEConfig,
    IndexingConfig,
    IngestionConfig,
    JinaEmbeddingConfig,
    LateChunkingConfig,
    MilvusConfig,
    OllamaLLMConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    PineconeConfig,
    PostRetrievalConfig,
    PreRetrievalConfig,
    PropositionChunkingConfig,
    QdrantConfig,
    RAGPipelineConfig,
    RecursiveChunkingConfig,
    SentenceWindowConfig,
    VoyageEmbeddingConfig,
)
from ragfactory.core.validator import (
    CostEstimate,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
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
