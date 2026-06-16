"""Unit tests for the RAGFactory FastAPI service."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

import ragfactory
from ragfactory.api.main import app, generate_pipeline, get_config_schema, validate_config


def test_app_version_matches_package_version() -> None:
    assert app.version == ragfactory.__version__


def test_schema_endpoint() -> None:
    schema = get_config_schema()
    assert schema["title"] == "RAGPipelineConfig"
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "indexing" in schema["properties"]


def test_validate_endpoint_valid() -> None:
    valid_config = {
        "name": "support-rag",
        "framework": "langchain",
        "indexing": {
            "chunking": {"type": "recursive"},
            "embedding": {"type": "openai"},
            "vector_db": {"type": "qdrant"},
        },
        "retrieval": {"type": "hybrid_rrf"},
        "generation": {"llm": {"type": "openai"}},
    }
    res = validate_config(valid_config)
    assert res["valid"] is True
    assert len(res["errors"]) == 0
    assert len(res["warnings"]) == 0


def test_validate_endpoint_invalid_schema() -> None:
    # invalid name pattern, missing indexing
    invalid_config = {
        "name": "SUPPORT_RAG",
    }
    with pytest.raises(HTTPException) as exc_info:
        validate_config(invalid_config)
    assert exc_info.value.status_code == 422


def test_validate_endpoint_incompatible() -> None:
    # hybrid_rrf with chromadb is incompatible
    incompat_config = {
        "name": "support-rag",
        "indexing": {
            "chunking": {"type": "recursive"},
            "embedding": {"type": "openai"},
            "vector_db": {"type": "chromadb"},
        },
        "retrieval": {"type": "hybrid_rrf"},
        "generation": {"llm": {"type": "openai"}},
    }
    res = validate_config(incompat_config)
    assert res["valid"] is False
    assert len(res["errors"]) > 0
    assert any("INCOMPAT_" in err["code"] for err in res["errors"])


def test_generate_endpoint_valid() -> None:
    valid_config = {
        "name": "support-rag",
        "framework": "langchain",
        "indexing": {
            "chunking": {"type": "recursive"},
            "embedding": {"type": "openai"},
            "vector_db": {"type": "chromadb"},
        },
        "retrieval": {"type": "dense"},
        "generation": {"llm": {"type": "openai"}},
    }
    response = generate_pipeline(valid_config)
    assert response.status_code == 200
    assert response.media_type == "application/zip"
    assert response.headers["content-disposition"] == "attachment; filename=support-rag.zip"

    # Verify zip content
    zip_bytes = response.body
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        files = zip_file.namelist()
        assert "pipeline.py" in files
        assert "ingestion.py" in files
        assert "api.py" in files
        assert "pyproject.toml" in files
        assert "config.yaml" in files

        pipeline_content = zip_file.read("pipeline.py").decode("utf-8")
        assert "run_pipeline" in pipeline_content
        # Verify return_contexts is present in generated signature
        assert "return_contexts: bool = False" in pipeline_content


def test_generate_endpoint_invalid() -> None:
    incompat_config = {
        "name": "support-rag",
        "indexing": {
            "chunking": {"type": "recursive"},
            "embedding": {"type": "openai"},
            "vector_db": {"type": "chromadb"},
        },
        "retrieval": {"type": "hybrid_rrf"},  # incompat
        "generation": {"llm": {"type": "openai"}},
    }
    with pytest.raises(HTTPException) as exc_info:
        generate_pipeline(incompat_config)
    assert exc_info.value.status_code == 400
