"""ragfactory CLI — generate, validate, and explore RAG pipeline configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional, cast

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

import ragfactory
from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.generator import GeneratorResult
from ragfactory.core.generator import generate as _gen
from ragfactory.core.validator import ValidationResult
from ragfactory.core.validator import validate as _val

# ─── App ──────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="ragfactory",
    help="Generate production-ready RAG pipelines from a single config.",
    add_completion=False,
    no_args_is_help=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────

_DENSE_ONLY_DBS: frozenset[str] = frozenset({"chromadb", "pinecone"})

# CLI accepts "cohere"; config discriminator requires "cohere_llm"
_LLM_TYPE_MAP: dict[str, str] = {
    "openai":    "openai",
    "anthropic": "anthropic",
    "cohere":    "cohere_llm",
    "ollama":    "ollama",
}

_COMPONENTS: dict[str, list[tuple[str, str]]] = {
    "chunking": [
        ("fixed",           "Fixed-size token windows. Simple baseline."),
        ("recursive",       "Hierarchical separators. Production default."),
        ("semantic",        "Embedding-based breakpoints. Handles topic drift."),
        ("contextual",      "Prepends LLM context to each chunk. Best recall (+49-67%)."),
        ("late",            "Jina Late Chunking. Token-level pooling after full encoding."),
        ("page_level",      "One chunk per PDF page. Good for structured docs."),
        ("proposition",     "Extracts atomic propositions before indexing. Higher cost."),
    ],
    "embedding": [
        ("openai",  "OpenAI text-embedding-3-small/large. Default 1536d."),
        ("cohere",  "Cohere embed-v3. 1024d. Multilingual."),
        ("voyage",  "Voyage AI. Best retrieval benchmarks (MTEB 2024)."),
        ("gemini",  "Google text-embedding-004. 768d."),
        ("bge_m3",  "BAAI/BGE-M3. Self-hosted. Dense+sparse+colbert."),
        ("nomic",   "Nomic embed-text. Self-hosted or API. 768d."),
        ("jina",    "Jina embeddings-v3. 1024d. Long context (8192 tokens)."),
    ],
    "vectordb": [
        ("chromadb", "Embedded, in-process. Prototyping only."),
        ("qdrant",   "Rust. Production default. 8500-12000 QPS. Best metadata filtering."),
        ("pinecone", "Serverless. Zero-ops. Up to 1.4B vectors."),
        ("weaviate", "Modular. Native hybrid. Multi-tenancy."),
        ("milvus",   "Distributed. GPU-accelerated (CAGRA). Billion-scale."),
        ("pgvector", "PostgreSQL extension. Best for existing Postgres stacks."),
    ],
    "retrieval": [
        ("dense",           "Pure vector similarity. Fast baseline."),
        ("hybrid_rrf",      "BM25 + dense via Reciprocal Rank Fusion. +15-30% recall."),
        ("hybrid_weighted", "BM25 + dense with tunable alpha weight."),
        ("small_to_big",    "Retrieve child chunks, return parent context."),
        ("sentence_window", "Retrieve sentence, expand to window."),
    ],
    "reranker": [
        ("cohere",        "Cohere Rerank API. Fast, no GPU needed."),
        ("cross_encoder", "Cross-encoder. Best quality, GPU recommended."),
        ("colbert",       "ColBERT (RAGatouille). Token-level interaction."),
        ("flashrank",     "FlashRank. Fastest local reranker. CPU-friendly."),
    ],
    "llm": [
        ("openai",    "GPT-4o / GPT-4o-mini. Default gpt-4o-mini."),
        ("anthropic", "Claude Sonnet/Opus/Haiku. Default claude-sonnet-4-6."),
        ("cohere",    "Command-R+. Grounding optimised."),
        ("ollama",    "Local inference. Zero API cost. Requires Ollama."),
    ],
}

_VALID_COMPONENTS: list[str] = list(_COMPONENTS.keys())
_FRAMEWORK_CHOICES: tuple[str, ...] = ("langchain", "llamaindex")
_CHUNKING_CHOICES: tuple[str, ...] = tuple(t for t, _ in _COMPONENTS["chunking"])
_EMBEDDING_CHOICES: tuple[str, ...] = tuple(t for t, _ in _COMPONENTS["embedding"])
_VECTOR_DB_CHOICES: tuple[str, ...] = tuple(t for t, _ in _COMPONENTS["vectordb"])
_RETRIEVAL_CHOICES: tuple[str, ...] = tuple(t for t, _ in _COMPONENTS["retrieval"])
_RERANKER_CHOICES: tuple[str, ...] = ("none", *(t for t, _ in _COMPONENTS["reranker"]))
_LLM_CHOICES: tuple[str, ...] = tuple(_LLM_TYPE_MAP.keys())

# ─── Version callback ─────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ragfactory {ragfactory.__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Generate production-ready RAG pipelines from a single config."""


def _choices_help(choices: tuple[str, ...]) -> str:
    return "|".join(choices)


def _validate_choice(
    value: str | None,
    choices: tuple[str, ...],
    label: str,
) -> str | None:
    if value is None or value in choices:
        return value
    raise typer.BadParameter(f"Invalid {label}. Choose from: {', '.join(choices)}")


def _framework_callback(value: str) -> str:
    return cast(str, _validate_choice(value, _FRAMEWORK_CHOICES, "framework"))


def _chunking_callback(value: str) -> str:
    return cast(str, _validate_choice(value, _CHUNKING_CHOICES, "chunking strategy"))


def _embedding_callback(value: str) -> str:
    return cast(str, _validate_choice(value, _EMBEDDING_CHOICES, "embedding model"))


def _vector_db_callback(value: str) -> str:
    return cast(str, _validate_choice(value, _VECTOR_DB_CHOICES, "vector database"))


def _retrieval_callback(value: str | None) -> str | None:
    return _validate_choice(value, _RETRIEVAL_CHOICES, "retrieval strategy")


def _reranker_callback(value: str) -> str:
    return cast(str, _validate_choice(value, _RERANKER_CHOICES, "reranker"))


def _llm_callback(value: str) -> str:
    return cast(str, _validate_choice(value, _LLM_CHOICES, "LLM provider"))


# ─── Private helpers ──────────────────────────────────────────────────────────


def _load_config(path: Path) -> RAGPipelineConfig:
    """Load a YAML or JSON pipeline config. Exits with code 1 on any failure."""
    if not path.exists():
        typer.echo(f"Error: Config file not found: {path}", err=True)
        raise typer.Exit(1)

    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            typer.echo(
                f"Error: Unsupported config format '{suffix}'. Use .yaml or .json",
                err=True,
            )
            raise typer.Exit(1)
    except yaml.YAMLError as exc:
        typer.echo(f"Error: Failed to parse YAML: {exc}", err=True)
        raise typer.Exit(1)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: Failed to parse JSON: {exc}", err=True)
        raise typer.Exit(1)

    try:
        return RAGPipelineConfig.model_validate(raw)
    except ValidationError as exc:
        typer.echo(f"Error: Invalid config:\n{exc}", err=True)
        raise typer.Exit(1)


def _resolve_retrieval(vector_db: str, retrieval: str | None) -> str:
    """Return retrieval type: honour explicit value or auto-select from vector_db."""
    if retrieval is not None:
        return retrieval
    return "dense" if vector_db in _DENSE_ONLY_DBS else "hybrid_rrf"


def _build_config_from_flags(
    name: str,
    framework: str,
    chunking: str,
    embedding: str,
    vector_db: str,
    retrieval: str | None,
    reranker: str,
    llm: str,
) -> RAGPipelineConfig:
    """Construct RAGPipelineConfig from init CLI flags.

    Important: ``retrieval`` is a top-level field on RAGPipelineConfig,
    NOT nested inside ``indexing``.
    """
    resolved_retrieval = _resolve_retrieval(vector_db, retrieval)
    # Build kwargs incrementally — omit post_retrieval when no reranker so Pydantic
    # uses its default_factory instead of receiving None and failing validation.
    kwargs: dict[str, object] = {
        "name":       name,
        "framework":  framework,
        "indexing": {
            "chunking":  {"type": chunking},
            "embedding": {"type": embedding},
            "vector_db": {"type": vector_db},
        },
        "retrieval":   {"type": resolved_retrieval},   # top-level, not inside indexing
        "generation":  {"llm": {"type": _LLM_TYPE_MAP[llm]}},
    }
    if reranker != "none":
        kwargs["post_retrieval"] = {"reranker": {"type": reranker}}
    try:
        return RAGPipelineConfig(**kwargs)
    except ValidationError as exc:
        typer.echo(f"Error: Invalid combination of flags:\n{exc}", err=True)
        raise typer.Exit(1)


def _write_output(
    files: dict[str, str],
    config_yaml: str,
    output_dir: Path,
) -> list[Path]:
    """Create output_dir, write generated files and config.yaml. Return written paths."""
    if output_dir.exists() and any(output_dir.iterdir()):
        typer.echo(
            f"Warning: Output directory '{output_dir}' is non-empty — files will be overwritten.",
            err=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, content in files.items():
        fp = output_dir / filename
        fp.write_text(content, encoding="utf-8")
        written.append(fp)
    config_path = output_dir / "config.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")
    written.append(config_path)
    return written


def _exit_on_generator_errors(result: GeneratorResult, console: Console) -> None:
    """Fail CLI commands when the generator returns an error result."""
    if result.validation_passed and not result.errors:
        return

    console.print("[red]Generation failed.[/red]")
    for error in result.errors:
        console.print(f"  [red]ERROR[/red]  {error}")
    raise typer.Exit(1)


def _print_validation(result: ValidationResult, console: Console) -> None:
    """Print errors (red), warnings (yellow), and infos (blue) via Rich."""
    for issue in result.errors:
        console.print(f"  [red]ERROR[/red]  [{issue.code}] {issue.message}")
    for issue in result.warnings:
        console.print(f"  [yellow]WARN [/yellow]  [{issue.code}] {issue.message}")
    for issue in result.infos:
        console.print(f"  [blue]INFO [/blue]   [{issue.code}] {issue.message}")


def _print_file_summary(written: list[Path], console: Console) -> None:
    """Print a Rich table summarising generated files."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("", justify="center")
    for fp in written:
        size = fp.stat().st_size
        table.add_row(fp.name, f"{size:,} B", "[green]OK[/green]")
    console.print(table)


# ─── Commands ─────────────────────────────────────────────────────────────────


@app.command(name="options")
def options_cmd(
    component: Annotated[
        Optional[str],
        typer.Option(
            "--component", "-c",
            help=f"Filter by component: {', '.join(_VALID_COMPONENTS)}",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List available components and their options."""
    console = Console()

    if component is not None and component not in _VALID_COMPONENTS:
        typer.echo(
            f"Error: Unknown component '{component}'. "
            f"Choose from: {', '.join(_VALID_COMPONENTS)}",
            err=True,
        )
        raise typer.Exit(1)

    if as_json:
        if component is not None:
            payload: object = [
                {"type": t, "description": d} for t, d in _COMPONENTS[component]
            ]
        else:
            payload = {
                k: [{"type": t, "description": d} for t, d in v]
                for k, v in _COMPONENTS.items()
            }
        typer.echo(json.dumps(payload, indent=2))
        return

    if component is not None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Description")
        for type_name, desc in _COMPONENTS[component]:
            table.add_row(type_name, desc)
        console.print(table)
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Component", style="bold magenta", no_wrap=True)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Description")
        for comp_name, entries in _COMPONENTS.items():
            for i, (type_name, desc) in enumerate(entries):
                table.add_row(comp_name if i == 0 else "", type_name, desc)
            table.add_section()
        console.print(table)


@app.command(name="validate")
def validate_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to pipeline config (.yaml or .json)"),
    ],
) -> None:
    """Validate a pipeline config for compatibility and correctness."""
    console = Console()
    cfg = _load_config(config)
    result = _val(cfg)
    _print_validation(result, console)
    if result.valid:
        console.print("[green]Valid[/green]")
        raise typer.Exit(0)
    n = len(result.errors)
    console.print(f"[red]Invalid — {n} error{'s' if n != 1 else ''}[/red]")
    raise typer.Exit(1)


@app.command(name="generate")
def generate_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to pipeline config (.yaml or .json)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory (default: ./{config.name})"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Generate even if validation errors exist."),
    ] = False,
) -> None:
    """Generate pipeline code from a config file."""
    console = Console()
    cfg = _load_config(config)

    val_result = _val(cfg)
    _print_validation(val_result, console)

    if val_result.errors and not force:
        n = len(val_result.errors)
        console.print(
            f"[red]{n} validation error{'s' if n != 1 else ''}."
            " Fix them or use --force to generate anyway.[/red]"
        )
        raise typer.Exit(1)

    gen_result = _gen(cfg)
    _exit_on_generator_errors(gen_result, console)
    out_dir = output if output is not None else Path(cfg.name.replace("/", "-"))
    written = _write_output(gen_result.files, gen_result.config_yaml, out_dir)

    console.print(f"\n[green]Generated {len(written)} files -> {out_dir}[/green]")
    _print_file_summary(written, console)


@app.command(name="init")
def init_cmd(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Pipeline name"),
    ],
    framework: Annotated[
        str,
        typer.Option(
            help=f"Framework: {_choices_help(_FRAMEWORK_CHOICES)}",
            callback=_framework_callback,
        ),
    ] = "langchain",
    chunking: Annotated[
        str,
        typer.Option(
            help=f"Chunking strategy: {_choices_help(_CHUNKING_CHOICES)}",
            callback=_chunking_callback,
        ),
    ] = "recursive",
    embedding: Annotated[
        str,
        typer.Option(
            help=f"Embedding model: {_choices_help(_EMBEDDING_CHOICES)}",
            callback=_embedding_callback,
        ),
    ] = "openai",
    vector_db: Annotated[
        str,
        typer.Option(
            "--vector-db",
            help=f"Vector database: {_choices_help(_VECTOR_DB_CHOICES)}",
            callback=_vector_db_callback,
        ),
    ] = "chromadb",
    retrieval: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Retrieval strategy (auto-selected if omitted): "
                f"{_choices_help(_RETRIEVAL_CHOICES)}"
            ),
            callback=_retrieval_callback,
        ),
    ] = None,
    reranker: Annotated[
        str,
        typer.Option(
            help=f"Reranker: {_choices_help(_RERANKER_CHOICES)}",
            callback=_reranker_callback,
        ),
    ] = "none",
    llm: Annotated[
        str,
        typer.Option(
            help=f"LLM provider: {_choices_help(_LLM_CHOICES)}",
            callback=_llm_callback,
        ),
    ] = "openai",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory (default: ./{name})"),
    ] = None,
    save_config: Annotated[
        Optional[Path],
        typer.Option("--save-config", help="Write config YAML to this path."),
    ] = None,
) -> None:
    """Create a new RAG pipeline from component flags."""
    console = Console()

    cfg = _build_config_from_flags(
        name=name,
        framework=framework,
        chunking=chunking,
        embedding=embedding,
        vector_db=vector_db,
        retrieval=retrieval,
        reranker=reranker,
        llm=llm,
    )

    val_result = _val(cfg)
    _print_validation(val_result, console)

    if val_result.errors:
        n = len(val_result.errors)
        console.print(f"[red]{n} validation error{'s' if n != 1 else ''}. Adjust your flags.[/red]")
        raise typer.Exit(1)

    # Write config YAML if --save-config is given (always, even if also generating)
    if save_config is not None:
        config_yaml_str = yaml.dump(
            cfg.model_dump(mode="json"),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        save_config.write_text(config_yaml_str, encoding="utf-8")
        console.print(f"[green]Config saved -> {save_config}[/green]")

    # Save-only mode: --save-config given, --output not given
    if save_config is not None and output is None:
        raise typer.Exit(0)

    # Generate code
    gen_result = _gen(cfg)
    _exit_on_generator_errors(gen_result, console)
    out_dir = output if output is not None else Path(name.replace("/", "-"))
    written = _write_output(gen_result.files, gen_result.config_yaml, out_dir)

    console.print(f"\n[green]Generated {len(written)} files -> {out_dir}[/green]")
    _print_file_summary(written, console)


@app.command(name="api")
def api_cmd(
    host: Annotated[
        str,
        typer.Option("--host", help="Host to bind the API server to."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port to bind the API server to."),
    ] = 8000,
) -> None:
    """Start the RAGFactory REST API server."""
    try:
        import uvicorn
        from ragfactory.api.main import app as api_app  # noqa: F401
    except ImportError:
        typer.echo(
            "Error: API dependencies not found.\n"
            "Please install ragfactory with the [api] extra:\n"
            "  pip install \"ragfactory[api]\"",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Starting RAGFactory API server at http://{host}:{port}")
    uvicorn.run("ragfactory.api.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
