"""Integration tests for the ragfactory CLI.

Design principles:
  - Uses typer.testing.CliRunner — real filesystem, real core, no mocking.
  - Every test asserting exit_code == 0 also asserts result.exception is None.
  - Tests that rely on the default output path (no --output) use
    monkeypatch.chdir(tmp_path) to avoid polluting the repo root.
  - All file paths in --output flags are absolute (str(tmp_path / ...)).
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragfactory.cli.main import app
from ragfactory.core.generator import GeneratorResult

runner = CliRunner()

# ─── Fixtures ─────────────────────────────────────────────────────────────────

_VALID_YAML = textwrap.dedent("""
    name: test-pipeline
    framework: langchain
    indexing:
      chunking: {type: recursive}
      embedding: {type: openai}
      vector_db: {type: chromadb}
    retrieval: {type: dense}
    generation:
      llm: {type: openai}
""").strip()

_INCOMPATIBLE_YAML = textwrap.dedent("""
    name: bad-pipeline
    framework: langchain
    indexing:
      chunking: {type: recursive}
      embedding: {type: openai}
      vector_db: {type: chromadb}
    retrieval: {type: hybrid_rrf}
    generation:
      llm: {type: openai}
""").strip()

# chromadb + dense: valid=True but triggers WARN_CHROMADB
_WARNING_YAML = textwrap.dedent("""
    name: warn-pipeline
    framework: langchain
    indexing:
      chunking: {type: recursive}
      embedding: {type: openai}
      vector_db: {type: chromadb}
    retrieval: {type: dense}
    generation:
      llm: {type: openai}
""").strip()


@pytest.fixture()
def valid_config(tmp_path: Path) -> Path:
    f = tmp_path / "valid.yaml"
    f.write_text(_VALID_YAML, encoding="utf-8")
    return f


@pytest.fixture()
def incompatible_config(tmp_path: Path) -> Path:
    f = tmp_path / "bad.yaml"
    f.write_text(_INCOMPATIBLE_YAML, encoding="utf-8")
    return f


@pytest.fixture()
def warning_config(tmp_path: Path) -> Path:
    f = tmp_path / "warn.yaml"
    f.write_text(_WARNING_YAML, encoding="utf-8")
    return f


# ─── generate ─────────────────────────────────────────────────────────────────

_EXPECTED_FILES = {
    "pipeline.py", "ingestion.py", "pyproject.toml",
    ".env.example", "README.md", "Dockerfile", "config.yaml",
}


class TestGenerate:
    def test_generate_creates_all_files(self, valid_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = runner.invoke(app, ["generate", "--config", str(valid_config), "--output", str(out)])
        assert result.exception is None
        assert result.exit_code == 0
        created = {p.name for p in out.iterdir()}
        assert created == _EXPECTED_FILES

    def test_generate_python_files_parse(self, valid_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = runner.invoke(app, ["generate", "--config", str(valid_config), "--output", str(out)])
        assert result.exception is None
        assert result.exit_code == 0
        for py_file in ("pipeline.py", "ingestion.py"):
            source = (out / py_file).read_text(encoding="utf-8")
            ast.parse(source)  # raises SyntaxError if generated code is broken

    def test_generate_config_yaml_written(self, valid_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = runner.invoke(app, ["generate", "--config", str(valid_config), "--output", str(out)])
        assert result.exception is None
        assert result.exit_code == 0
        config_file = out / "config.yaml"
        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert data["name"] == "test-pipeline"

    def test_generate_default_output_dir(
        self, valid_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No --output → uses config.name as dir name relative to CWD
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["generate", "--config", str(valid_config)])
        assert result.exception is None
        assert result.exit_code == 0
        # config.name is "test-pipeline"
        assert (tmp_path / "test-pipeline").is_dir()
        assert (tmp_path / "test-pipeline" / "pipeline.py").exists()

    def test_generate_invalid_config_exits_1(
        self, incompatible_config: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["generate", "--config", str(incompatible_config), "--output", str(out)]
        )
        assert result.exit_code == 1
        assert not out.exists()  # no files written

    def test_generate_force_skips_validation(
        self, incompatible_config: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            ["generate", "--config", str(incompatible_config), "--output", str(out), "--force"],
        )
        assert result.exception is None
        assert result.exit_code == 0
        assert (out / "pipeline.py").exists()

    def test_generate_missing_file_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["generate", "--config", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_generate_generator_failure_exits_1(
        self,
        valid_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "out"
        monkeypatch.setattr(
            "ragfactory.cli.main._gen",
            lambda cfg: GeneratorResult(
                validation_passed=False,
                errors=["synthetic generator failure"],
                config_yaml="name: broken\n",
            ),
        )
        result = runner.invoke(app, ["generate", "--config", str(valid_config), "--output", str(out)])
        assert result.exit_code == 1
        assert "synthetic generator failure" in result.output
        assert not out.exists()


# ─── validate ─────────────────────────────────────────────────────────────────


class TestValidate:
    def test_validate_valid_exits_0(self, valid_config: Path) -> None:
        result = runner.invoke(app, ["validate", "--config", str(valid_config)])
        assert result.exception is None
        assert result.exit_code == 0
        assert "Valid" in result.output

    def test_validate_invalid_exits_1(self, incompatible_config: Path) -> None:
        result = runner.invoke(app, ["validate", "--config", str(incompatible_config)])
        assert result.exit_code == 1
        assert "INCOMPAT_HYBRID_RRF_CHROMADB" in result.output

    def test_validate_shows_warnings(self, warning_config: Path) -> None:
        # Valid config with chromadb warning — must exit 0 (warnings don't block)
        result = runner.invoke(app, ["validate", "--config", str(warning_config)])
        assert result.exception is None
        assert result.exit_code == 0
        assert "WARN_CHROMADB" in result.output


# ─── options ──────────────────────────────────────────────────────────────────


class TestOptions:
    def test_options_all_components(self) -> None:
        result = runner.invoke(app, ["options"])
        assert result.exception is None
        assert result.exit_code == 0
        for component in ("chunking", "embedding", "vectordb", "retrieval", "reranker", "llm"):
            assert component in result.output

    def test_options_filter_component(self) -> None:
        result = runner.invoke(app, ["options", "--component", "embedding"])
        assert result.exception is None
        assert result.exit_code == 0
        assert "openai" in result.output
        assert "voyage" in result.output
        assert "chromadb" not in result.output  # vectordb entry must not bleed in

    def test_options_json_output(self) -> None:
        result = runner.invoke(app, ["options", "--json"])
        assert result.exception is None
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "chunking" in data
        assert "llm" in data
        assert isinstance(data["embedding"], list)
        assert data["embedding"][0]["type"] == "openai"

    def test_options_unknown_component_exits_1(self) -> None:
        result = runner.invoke(app, ["options", "--component", "nonexistent"])
        assert result.exit_code == 1

    def test_options_chunking_lists_proposition_not_sentence_window(self) -> None:
        result = runner.invoke(app, ["options", "--component", "chunking"])
        assert result.exception is None
        assert result.exit_code == 0
        assert "proposition" in result.output
        assert "sentence_window" not in result.output


# ─── init ─────────────────────────────────────────────────────────────────────


class TestInit:
    def test_init_creates_output(self, tmp_path: Path) -> None:
        out = tmp_path / "foo"
        result = runner.invoke(
            app,
            ["init", "--name", "foo", "--vector-db", "chromadb", "--llm", "openai",
             "--output", str(out)],
        )
        assert result.exception is None
        assert result.exit_code == 0
        assert (out / "pipeline.py").exists()

    def test_init_save_config_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # --save-config only (no --output) → write YAML, skip code generation
        monkeypatch.chdir(tmp_path)
        save_path = tmp_path / "foo.yaml"
        result = runner.invoke(
            app,
            ["init", "--name", "foo", "--save-config", str(save_path)],
        )
        assert result.exception is None
        assert result.exit_code == 0
        assert save_path.exists()
        data = yaml.safe_load(save_path.read_text(encoding="utf-8"))
        assert data["name"] == "foo"
        # No output directory created
        assert not (tmp_path / "foo").exists()

    def test_init_auto_retrieval_chromadb(self, tmp_path: Path) -> None:
        # chromadb → auto-select dense (no BM25)
        out = tmp_path / "dense-test"
        result = runner.invoke(
            app,
            ["init", "--name", "dense-test", "--vector-db", "chromadb",
             "--output", str(out)],
        )
        assert result.exception is None
        assert result.exit_code == 0
        pipeline_src = (out / "pipeline.py").read_text(encoding="utf-8")
        assert "BM25" not in pipeline_src
        assert "EnsembleRetriever" not in pipeline_src

    def test_init_auto_retrieval_qdrant(self, tmp_path: Path) -> None:
        # qdrant → auto-select hybrid_rrf (contains BM25)
        out = tmp_path / "hybrid-test"
        result = runner.invoke(
            app,
            ["init", "--name", "hybrid-test", "--vector-db", "qdrant",
             "--output", str(out)],
        )
        assert result.exception is None
        assert result.exit_code == 0
        pipeline_src = (out / "pipeline.py").read_text(encoding="utf-8")
        assert "BM25" in pipeline_src


# ─── --help / --version ───────────────────────────────────────────────────────

class TestInitAdditional:
    def test_init_accepts_proposition_chunking(self, tmp_path: Path) -> None:
        out = tmp_path / "proposition-test"
        result = runner.invoke(
            app,
            [
                "init",
                "--name",
                "proposition-test",
                "--chunking",
                "proposition",
                "--vector-db",
                "chromadb",
                "--output",
                str(out),
            ],
        )
        assert result.exception is None
        assert result.exit_code == 0
        assert (out / "pipeline.py").exists()

    def test_init_rejects_sentence_window_chunking(self, tmp_path: Path) -> None:
        out = tmp_path / "bad-chunking"
        result = runner.invoke(
            app,
            [
                "init",
                "--name",
                "bad-chunking",
                "--chunking",
                "sentence_window",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "sentence_window" in result.output
        assert "Invalid value" in result.output
        assert not out.exists()

    def test_init_generator_failure_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "broken-init"
        monkeypatch.setattr(
            "ragfactory.cli.main._gen",
            lambda cfg: GeneratorResult(
                validation_passed=False,
                errors=["synthetic init generator failure"],
                config_yaml="name: broken\n",
            ),
        )
        result = runner.invoke(
            app,
            ["init", "--name", "broken-init", "--output", str(out)],
        )
        assert result.exit_code == 1
        assert "synthetic init generator failure" in result.output
        assert not out.exists()


class TestMeta:
    def test_help_exits_0(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exception is None
        assert result.exit_code == 0
        for cmd in ("generate", "validate", "init", "options"):
            assert cmd in result.output

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exception is None
        assert result.exit_code == 0
        assert "ragfactory" in result.output
        # version string contains a digit
        assert any(ch.isdigit() for ch in result.output)
