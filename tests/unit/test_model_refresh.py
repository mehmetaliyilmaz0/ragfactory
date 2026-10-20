from __future__ import annotations

import ast

import pytest
from pydantic import ValidationError

from ragfactory.core.config import (
    AnthropicLLMConfig,
    CohereEmbeddingConfig,
    CohereLLMConfig,
    EvaluationConfig,
    GeminiEmbeddingConfig,
    GenerationConfig,
    IndexingConfig,
    OpenAIEmbeddingConfig,
    OpenAILLMConfig,
    QdrantConfig,
    RAGPipelineConfig,
    VoyageEmbeddingConfig,
)
from ragfactory.core.generator import GeneratorError, _get_embedding_dim, generate


def _pipeline(embedding: object) -> RAGPipelineConfig:
    return RAGPipelineConfig(
        name="model-test",
        indexing=IndexingConfig(embedding=embedding, vector_db=QdrantConfig()),  # type: ignore[arg-type]
        generation=GenerationConfig(llm=OpenAILLMConfig()),
    )


@pytest.mark.parametrize(
    ("config", "model"),
    [
        (OpenAILLMConfig, "gpt-future-model"),
        (AnthropicLLMConfig, "claude-future-model"),
        (CohereLLMConfig, "command-future-model"),
    ],
)
def test_llm_model_fields_allow_forward_compatible_ids(config: type, model: str) -> None:
    assert config(model=model).model == model


@pytest.mark.parametrize("config", [OpenAILLMConfig, AnthropicLLMConfig, CohereLLMConfig])
def test_llm_model_fields_reject_empty_ids(config: type) -> None:
    with pytest.raises(ValidationError):
        config(model=" ")


@pytest.mark.parametrize(
    ("config", "model", "dimension"),
    [
        (OpenAIEmbeddingConfig, "text-embedding-3-small", 1536),
        (CohereEmbeddingConfig, "embed-v5.0-fast", 2048),
        (VoyageEmbeddingConfig, "voyage-4-large", 1024),
        (VoyageEmbeddingConfig, "voyage-3-lite", 512),
        (GeminiEmbeddingConfig, "gemini-embedding-2", 3072),
    ],
)
def test_current_embedding_models_have_documented_dimensions(
    config: type, model: str, dimension: int
) -> None:
    embedding = config(model=model)
    assert _get_embedding_dim(_pipeline(embedding)) == dimension


@pytest.mark.parametrize("model", ["text-embedding-004", "embedding-001"])
def test_retired_google_embedding_models_fail_with_migration_guidance(model: str) -> None:
    with pytest.raises(ValidationError, match="gemini-embedding-2"):
        GeminiEmbeddingConfig(model=model)


def test_unknown_embedding_dimension_fails_instead_of_guessing() -> None:
    embedding = OpenAIEmbeddingConfig(model="text-embedding-future")
    with pytest.raises(GeneratorError, match="Unsupported embedding model"):
        _get_embedding_dim(_pipeline(embedding))


def test_custom_llm_model_id_is_rendered_as_a_safe_python_literal() -> None:
    model = 'gpt-custom-"\nraise RuntimeError("injected") #'
    config = _pipeline(OpenAIEmbeddingConfig())
    config.generation.llm = OpenAILLMConfig(model=model)

    result = generate(config)
    pipeline = result.files["pipeline.py"]
    ast.parse(pipeline)
    calls = [node for node in ast.walk(ast.parse(pipeline)) if isinstance(node, ast.Call)]
    model_arguments = [
        keyword.value.value
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "ChatOpenAI"
        for keyword in call.keywords
        if keyword.arg == "model" and isinstance(keyword.value, ast.Constant)
    ]
    assert model_arguments == [model]


def test_evaluation_metrics_reject_unknown_provider_names() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(metrics=["unknown-metric"])


def test_evaluation_judge_model_is_forward_compatible() -> None:
    assert EvaluationConfig(judge_model="gpt-future-model").judge_model == "gpt-future-model"
