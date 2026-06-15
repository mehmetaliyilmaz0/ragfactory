"""RAGFactory REST API — Programmatic validation, generation and JSON schema endpoints."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.generator import generate as core_generate
from ragfactory.core.validator import validate as core_validate

app = FastAPI(
    title="RAGFactory API",
    description="Programmatic RAG pipeline generation and validation engine.",
    version="0.1.0",
)


@app.post("/api/v1/validate", tags=["validation"])
def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a RAG pipeline configuration, returning compatibility issues and cost estimates."""
    try:
        cfg = RAGPipelineConfig.model_validate(config)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=json.loads(e.json()),
        ) from e

    res = core_validate(cfg)
    return {
        "valid": res.valid,
        "errors": [
            {
                "code": err.code,
                "message": err.message,
                "component_path": err.component_path,
                "suggestion": err.suggestion,
            }
            for err in res.errors
        ],
        "warnings": [
            {
                "code": warn.code,
                "message": warn.message,
                "component_path": warn.component_path,
                "suggestion": warn.suggestion,
            }
            for warn in res.warnings
        ],
        "infos": [
            {
                "code": info.code,
                "message": info.message,
                "component_path": info.component_path,
                "suggestion": info.suggestion,
            }
            for info in res.infos
        ],
        "costs": [
            {
                "component": c.component,
                "description": c.description,
                "cost_per_million_tokens": c.cost_per_million_tokens,
                "estimated_total": c.estimated_total,
            }
            for c in res.costs
        ],
    }


@app.post("/api/v1/generate", tags=["generation"])
def generate_pipeline(config: dict[str, Any]) -> StreamingResponse:
    """Generate and return a ZIP archive containing the scaffolded RAG pipeline project."""
    try:
        cfg = RAGPipelineConfig.model_validate(config)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=json.loads(e.json()),
        ) from e

    val_res = core_validate(cfg)
    if not val_res.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Configuration has validation errors",
                "errors": [err.message for err in val_res.errors],
            },
        )

    gen_res = core_generate(cfg)
    if not gen_res.validation_passed:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Template generation generated invalid Python syntax or failed",
                "errors": gen_res.errors,
            },
        )

    # Pack generated files in-memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path, content in gen_res.files.items():
            zip_file.writestr(path, content)
        # Write the serialized clean config YAML
        zip_file.writestr("config.yaml", gen_res.config_yaml)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={cfg.name}.zip"},
    )


@app.get("/api/v1/schema", tags=["schema"])
def get_config_schema() -> dict[str, Any]:
    """Return the JSON Schema of RAGPipelineConfig to drive dynamic front-end form generation."""
    return RAGPipelineConfig.model_json_schema()
