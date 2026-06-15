"""Unit tests for the RAGFactory FastAPI service."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from ragfactory.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_schema_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    schema = response.json()
    assert schema["title"] == "RAGPipelineConfig"
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "indexing" in schema["properties"]


def test_validate_endpoint_valid(client: TestClient) -> None:
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
    response = client.post("/api/v1/validate", json=valid_config)
    assert response.status_code == 200
    res = response.json()
    assert res["valid"] is True
    assert len(res["errors"]) == 0
    assert len(res["warnings"]) == 0


def test_validate_endpoint_invalid_schema(client: TestClient) -> None:
    # invalid name pattern, missing indexing
    invalid_config = {
        "name": "SUPPORT_RAG",
    }
    response = client.post("/api/v1/validate", json=invalid_config)
    assert response.status_code == 422


def test_validate_endpoint_incompatible(client: TestClient) -> None:
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
    response = client.post("/api/v1/validate", json=incompat_config)
    assert response.status_code == 200
    res = response.json()
    assert res["valid"] is False
    assert len(res["errors"]) > 0
    assert any("INCOMPAT_" in err["code"] for err in res["errors"])


def test_generate_endpoint_valid(client: TestClient) -> None:
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
    response = client.post("/api/v1/generate", json=valid_config)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    # Verify zip content
    zip_bytes = response.content
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


def test_generate_endpoint_invalid(client: TestClient) -> None:
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
    response = client.post("/api/v1/generate", json=incompat_config)
    assert response.status_code == 400
