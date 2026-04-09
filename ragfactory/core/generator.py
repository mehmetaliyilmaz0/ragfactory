"""
Jinja2-based code generator for RAGFactory pipeline configurations.

Design principles:
  - Single public function: generate(config) -> GeneratorResult
  - Pure function — no side effects, no disk writes, no I/O
  - CLI (Phase 1d) and API (Phase 2) handle writing; generator only renders
  - StrictUndefined: templates must declare every variable they use
  - ast.parse() check on every generated .py file — template bugs surface loudly
  - Stage-first architecture: N templates (one per component), not N×M combinations

Template layout (relative to ragfactory/templates/):
  stages/
    chunking/<type>.py.j2
    embedding/<type>.py.j2
    vectordb/<type>.py.j2
    retrieval/<type>.py.j2
    reranker/<type>.py.j2
    llm/<type>.py.j2
  entrypoints/
    langchain/pipeline.py.j2
    langchain/ingestion.py.j2
    llamaindex/pipeline.py.j2
    llamaindex/ingestion.py.j2
    common/pyproject.toml.j2
    common/.env.example.j2
    common/README.md.j2
    common/Dockerfile.j2
    common/docker-compose.yml.j2  (only for external-service vector DBs)
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from importlib.resources import files as _pkg_files
from pathlib import Path

import jinja2

from ragfactory.core._providers import infer_context_model_provider
from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.versions import get_dependencies

# Default template directory shipped with the package.
# Path(str(...)) converts the importlib Traversable to a real Path, which
# jinja2.FileSystemLoader requires. This also works in zip-wheel installs
# where Path(__file__).parent would fail.
_DEFAULT_TEMPLATE_DIR = Path(str(_pkg_files("ragfactory") / "templates"))

# Vector DBs that require an external service → include docker-compose.yml
_EXTERNAL_SERVICE_DBS = {"qdrant", "weaviate", "milvus", "pgvector"}

# API key env vars needed per component type
_ENV_VAR_MAP: dict[str, list[tuple[str, str]]] = {
    # embedding
    "openai":    [("OPENAI_API_KEY", "OpenAI API key")],
    "cohere":    [("COHERE_API_KEY", "Cohere API key")],
    "voyage":    [("VOYAGE_API_KEY", "Voyage AI API key")],
    "gemini":    [("GOOGLE_API_KEY", "Google AI API key")],
    "jina":      [("JINA_API_KEY", "Jina AI API key")],
    # vector db
    "qdrant":    [("QDRANT_API_KEY", "Qdrant API key (optional for local)")],
    "pinecone":  [("PINECONE_API_KEY", "Pinecone API key")],
    "weaviate":  [("WEAVIATE_API_KEY", "Weaviate API key (optional for local)")],
    "milvus":    [("MILVUS_TOKEN", "Zilliz Cloud token (optional for local Milvus)")],
    "pgvector":  [("PGVECTOR_CONNECTION_STRING", "PostgreSQL connection string")],
    # llm
    "anthropic": [("ANTHROPIC_API_KEY", "Anthropic API key")],
    "cohere_llm":[("COHERE_API_KEY", "Cohere API key")],
    # web search (CRAG)
    "tavily":    [("TAVILY_API_KEY", "Tavily Search API key")],
    "serper":    [("SERPER_API_KEY", "Serper API key")],
    # parsers
    "azure_doc_intelligence": [
        ("AZURE_DOC_INTELLIGENCE_KEY", "Azure Document Intelligence key"),
        ("AZURE_DOC_INTELLIGENCE_ENDPOINT", "Azure Document Intelligence endpoint"),
    ],
}


# ─── Output types ─────────────────────────────────────────────────────────────


@dataclass
class GeneratedFile:
    path:      str   # relative path, e.g. "pipeline.py"
    content:   str
    is_python: bool  # True → ast.parse() is run on content


@dataclass
class GeneratorResult:
    """
    Result of a generate() call.

    .generated_files    — canonical list of GeneratedFile (source of truth)
    .files              — convenience dict[path → content], computed from generated_files
    .validation_passed  — False if any ast.parse() error occurred
    .errors             — list of ast.parse error strings (empty on success)
    .config_yaml        — the input config round-tripped to YAML
    """

    generated_files:   list[GeneratedFile] = field(default_factory=list)
    validation_passed: bool = True
    errors:            list[str] = field(default_factory=list)
    config_yaml:       str = ""

    @property
    def files(self) -> dict[str, str]:
        """Convenience dict built from generated_files. Read-only."""
        return {gf.path: gf.content for gf in self.generated_files}


class GeneratorError(Exception):
    """Raised when code generation fails due to a template or engine bug."""


# ─── Template loader ──────────────────────────────────────────────────────────


class TemplateLoader:
    """
    Loads and renders Jinja2 templates from a template directory.

    Uses StrictUndefined so any template referencing an undefined variable
    fails immediately with a clear error rather than silently rendering empty.
    """

    def __init__(self, template_dir: Path | None = None) -> None:
        self._template_dir = template_dir or _DEFAULT_TEMPLATE_DIR
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._template_dir)),
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # tojson_py: serialise a Python value to a JSON literal suitable for
        # embedding inside generated .py files. Unlike Jinja2's built-in tojson,
        # this filter does NOT escape HTML entities (<, >, &, '), so the output
        # is a clean Python string literal rather than HTML-safe JSON.
        self._env.filters["tojson_py"] = lambda v: json.dumps(v, ensure_ascii=False)

    def _render(self, template_path: str, ctx: dict) -> str:  # noqa: ANN001
        """Load and render a template by path. Single point for all error handling."""
        try:
            return self._env.get_template(template_path).render(ctx)
        except jinja2.TemplateNotFound:
            raise GeneratorError(
                f"Template not found: {template_path}\n"
                "This is a bug in ragfactory. Please report it at "
                "https://github.com/mehmetaliyilmaz0/ragfactory/issues"
            ) from None
        except jinja2.UndefinedError as e:
            raise GeneratorError(
                f"Template variable error in {template_path}: {e}\n"
                "This is a bug in ragfactory templates."
            ) from e

    def render_stage(self, category: str, type_name: str, ctx: dict) -> str:  # noqa: ANN001
        """Render stages/<category>/<type_name>.py.j2"""
        return self._render(f"stages/{category}/{type_name}.py.j2", ctx)

    def render_entrypoint(self, framework: str, name: str, ctx: dict) -> str:  # noqa: ANN001
        """Render entrypoints/<framework>/<name>.py.j2"""
        return self._render(f"entrypoints/{framework}/{name}.py.j2", ctx)

    def render_common(self, name: str, ctx: dict) -> str:  # noqa: ANN001
        """Render entrypoints/common/<name>.j2"""
        return self._render(f"entrypoints/common/{name}.j2", ctx)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _validate_python(code: str, filename: str) -> list[str]:
    """Return a list of syntax error strings. Empty list means valid."""
    try:
        ast.parse(code, filename=filename)
        return []
    except SyntaxError as e:
        return [
            f"{filename}:{e.lineno}: {e.msg}\n"
            f"  This is a bug in ragfactory templates. Please report it.\n"
            f"  Context: {e.text!r}"
        ]


def _collect_required_env_vars(config: RAGPipelineConfig) -> list[tuple[str, str]]:
    """
    Return a deduplicated list of (ENV_VAR_NAME, description) pairs
    for all API keys required by this config.
    """
    seen: set[str] = set()
    result: list[tuple[str, str]] = []

    def _add(key: str) -> None:
        for entry in _ENV_VAR_MAP.get(key, []):
            if entry[0] not in seen:
                seen.add(entry[0])
                result.append(entry)

    # Embedding
    _add(config.indexing.embedding.type)
    # Vector DB
    _add(config.indexing.vector_db.type)
    # LLM — OpenAI LLM shares OPENAI_API_KEY with openai embedding;
    # _add deduplicates via `seen`, so no special case needed.
    _add(config.generation.llm.type)

    # Contextual chunking may need an extra API key for the context model
    if config.indexing.chunking.type == "contextual":
        ctx_model: str = config.indexing.chunking.context_model  # type: ignore[union-attr]
        provider = infer_context_model_provider(ctx_model)
        if provider is not None:
            _add(provider)

    # CRAG web search
    adv = config.generation.advanced
    if adv is not None and adv.crag is not None and adv.crag.enabled:
        _add(adv.crag.web_search_provider)

    # Parser
    if config.ingestion.parser == "azure_doc_intelligence":
        _add("azure_doc_intelligence")

    return result


# ─── Embedding Dimension Lookup ───────────────────────────────────────────────

_EMBEDDING_DIMS: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Cohere
    "embed-v4.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    # Voyage
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-finance-2": 1024,
    "voyage-code-2": 1536,
    "voyage-large-2": 1536,
    # Gemini
    "models/text-embedding-004": 768,
    # BGE-M3
    "BAAI/bge-m3": 1024,
    # Nomic
    "nomic-embed-text-v1.5": 768,
    "nomic-embed-text-v1": 768,
    # Jina
    "jina-embeddings-v3": 1024,
    "jina-embeddings-v2-base-en": 768,
}


_EMBEDDING_TYPE_DIMS: dict[str, int] = {
    "bge_m3": 1024,
    "nomic": 768,
    "jina": 1024,
}


def _get_embedding_dim(config: "RAGPipelineConfig") -> int:
    """Return the output vector dimension for the configured embedding model."""
    emb = config.indexing.embedding
    # OpenAIEmbeddingConfig has an explicit dimensions field
    if hasattr(emb, "dimensions") and emb.dimensions is not None:
        return emb.dimensions
    # Configs with a model field (OpenAI, Cohere, Voyage, Gemini, Jina, Nomic)
    if hasattr(emb, "model") and emb.model is not None:
        return _EMBEDDING_DIMS.get(emb.model, 1536)
    # Configs identified by type only (BGE-M3, HuggingFace-based, etc.)
    return _EMBEDDING_TYPE_DIMS.get(str(emb.type), 1536)


# ─── Generator stages ─────────────────────────────────────────────────────────


def _framework_str(config: RAGPipelineConfig) -> str:
    """Return framework as a plain str path segment for template path construction."""
    # use_enum_values=True on StrictModel guarantees config.framework is already
    # a plain str at runtime. The hasattr/.value branch is defensive-only and will
    # not fire under normal operation, but guards against future config schema changes.
    fw = config.framework
    return fw.value if hasattr(fw, "value") else str(fw)  # type: ignore[union-attr]


def _render_stages(
    config: RAGPipelineConfig,
    loader: TemplateLoader,
    fw: str,
) -> dict[str, str]:
    """
    Render all stage templates and return a dict of stage_name → rendered_code.
    Stages are rendered in data-flow order.

    Template context contract:
      - Per-stage templates (stages/<cat>/<type>.py.j2) should read the
        unpacked sub-object keys ("chunking", "embedding", "vector_db",
        "retrieval", "reranker", "llm") and NOT reach into `config`.
      - Entrypoint templates (pipeline.py.j2, ingestion.py.j2) read `config`
        for cross-cutting concerns (name, framework, dependencies).
      - `config` is passed to stages only for rare cross-cutting reads and
        should be used sparingly — prefer the specific sub-object key.
    """
    base_ctx: dict = {
        "config":        config,
        "framework":     fw,
        "pipeline_name": config.name,
        "dependencies":  get_dependencies(config),
        "python_version": "3.11",
    }

    stages: dict[str, str] = {}

    # 1. Chunking
    stages["chunking"] = loader.render_stage(
        "chunking",
        config.indexing.chunking.type,
        {**base_ctx, "chunking": config.indexing.chunking},
    )

    # 2. Embedding
    stages["embedding"] = loader.render_stage(
        "embedding",
        config.indexing.embedding.type,
        {
            **base_ctx,
            "embedding": config.indexing.embedding,
            "is_late_chunking": config.indexing.chunking.type == "late",
        },
    )

    # 3. Vector DB
    stages["vectordb"] = loader.render_stage(
        "vectordb",
        config.indexing.vector_db.type,
        {**base_ctx, "vector_db": config.indexing.vector_db, "embedding_dim": _get_embedding_dim(config)},
    )

    # 4. Retrieval
    stages["retrieval"] = loader.render_stage(
        "retrieval",
        config.retrieval.type,  # type: ignore[union-attr]
        {
            **base_ctx,
            "retrieval": config.retrieval,
            "vector_db_type": config.indexing.vector_db.type,
        },
    )

    # 5. Reranker (optional — skip entirely if None)
    if config.post_retrieval.reranker is not None:
        stages["reranker"] = loader.render_stage(
            "reranker",
            config.post_retrieval.reranker.type,
            {**base_ctx, "reranker": config.post_retrieval.reranker},
        )

    # 6. LLM
    stages["llm"] = loader.render_stage(
        "llm",
        config.generation.llm.type,
        {
            **base_ctx,
            "llm": config.generation.llm,
            "prompt_template": config.generation.prompt_template,
            "advanced": config.generation.advanced,
        },
    )

    return stages


# ─── Public API ───────────────────────────────────────────────────────────────


def generate(
    config: RAGPipelineConfig,
    template_dir: Path | None = None,
) -> GeneratorResult:
    """
    Generate pipeline code from a validated RAGPipelineConfig.

    This is a pure function — it never writes to disk.
    The caller (CLI / API) is responsible for persisting the output.

    Args:
        config:       A validated RAGPipelineConfig instance.
        template_dir: Override the default template directory.
                      Primarily used for testing with stub templates.

    Returns:
        GeneratorResult with all generated files and any syntax errors.
        .validation_passed is False if any .py file has a syntax error.
    """
    loader = TemplateLoader(template_dir)
    generated_files: list[GeneratedFile] = []
    all_errors: list[str] = []
    fw = _framework_str(config)

    base_ctx: dict = {
        "config":        config,
        "framework":     fw,
        "pipeline_name": config.name,
        "dependencies":  get_dependencies(config),
        "python_version": "3.11",
    }

    try:
        # ── Render stages ────────────────────────────────────────────────────
        stages = _render_stages(config, loader, fw)

        # ── Entrypoint context ───────────────────────────────────────────────
        entrypoint_ctx: dict = {
            **base_ctx,
            "stages":       stages,
            "pre_retrieval":  config.pre_retrieval,
            "post_retrieval": config.post_retrieval,
            "generation":     config.generation,
            "ingestion":      config.ingestion,
        }

        # ── Python entrypoints ───────────────────────────────────────────────
        pipeline_py  = loader.render_entrypoint(fw, "pipeline",  entrypoint_ctx)
        ingestion_py = loader.render_entrypoint(fw, "ingestion", entrypoint_ctx)

        # ── Common files ─────────────────────────────────────────────────────
        env_vars = _collect_required_env_vars(config)
        common_ctx: dict = {**base_ctx, "env_vars": env_vars}

        pyproject_toml = loader.render_common("pyproject.toml", common_ctx)
        env_example    = loader.render_common(".env.example",   common_ctx)
        readme_md      = loader.render_common("README.md",      common_ctx)
        dockerfile     = loader.render_common("Dockerfile",     common_ctx)

        # docker-compose.yml only for external-service vector DBs
        docker_compose: str | None = None
        if config.indexing.vector_db.type in _EXTERNAL_SERVICE_DBS:
            docker_compose = loader.render_common(
                "docker-compose.yml",
                {**base_ctx, "vector_db": config.indexing.vector_db},
            )

    except GeneratorError as e:
        # Template / engine error — return immediately with error state
        return GeneratorResult(
            generated_files=[],
            validation_passed=False,
            errors=[str(e)],
            config_yaml=config.to_yaml(),
        )

    # ── Build file list ───────────────────────────────────────────────────────
    py_files: list[tuple[str, str]] = [
        ("pipeline.py",  pipeline_py),
        ("ingestion.py", ingestion_py),
    ]
    non_py_files: list[tuple[str, str]] = [
        ("pyproject.toml", pyproject_toml),
        (".env.example",   env_example),
        ("README.md",      readme_md),
        ("Dockerfile",     dockerfile),
    ]
    if docker_compose is not None:
        non_py_files.append(("docker-compose.yml", docker_compose))

    # ── ast.parse() all Python files ─────────────────────────────────────────
    for path, content in py_files:
        errors = _validate_python(content, path)
        all_errors.extend(errors)
        generated_files.append(GeneratedFile(path=path, content=content, is_python=True))

    for path, content in non_py_files:
        generated_files.append(GeneratedFile(path=path, content=content, is_python=False))

    return GeneratorResult(
        generated_files=generated_files,
        validation_passed=len(all_errors) == 0,
        errors=all_errors,
        config_yaml=config.to_yaml(),
    )
