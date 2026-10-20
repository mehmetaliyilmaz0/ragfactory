from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.generator import TemplateLoader


def _ingestion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, glob: str):
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    documents = ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    documents.Document = Document
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents)
    config = RAGPipelineConfig.model_validate(
        {
            "name": "review-regression",
            "ingestion": {"sources": [{"type": "file", "path": "docs", "glob": glob}]},
            "indexing": {"embedding": {"type": "openai"}, "vector_db": {"type": "qdrant"}},
            "generation": {"llm": {"type": "openai"}},
        }
    )
    source = TemplateLoader().render_entrypoint(
        "langchain",
        "ingestion",
        {
            "pipeline_name": config.name,
            "ingestion": config.ingestion,
            "stages": {"chunking": "", "embedding": "", "vectordb": ""},
        },
    )
    module = ModuleType("review_ingestion")
    module.__file__ = str(tmp_path / "ingestion.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    stored = []
    module.build_chunking = lambda: type("Chunker", (), {"split_documents": lambda _, docs: docs})()
    module.build_embedding = lambda: None
    module.build_vectordb = lambda _: type(
        "Store", (), {"add_documents": lambda _, docs: stored.extend(docs)}
    )()
    return module, stored


def test_explicit_all_files_glob_indexes_markdown_and_text(monkeypatch, tmp_path):
    module, stored = _ingestion(monkeypatch, tmp_path, "**/*")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("markdown", encoding="utf-8")
    (docs / "b.txt").write_text("plain text", encoding="utf-8")
    module.ingest(str(docs))
    assert sorted(doc.page_content for doc in stored) == ["markdown", "plain text"]
    corpus = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert sorted(doc["page_content"] for doc in corpus) == ["markdown", "plain text"]


@pytest.mark.parametrize("directory", [True, False])
def test_unsupported_binary_format_fails_before_indexing(monkeypatch, tmp_path, directory):
    module, stored = _ingestion(monkeypatch, tmp_path, "**/*")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("must not be partially indexed", encoding="utf-8")
    binary = docs / "b.pdf"
    binary.write_bytes(b"%PDF-1.7\n\xff\x00")
    with pytest.raises(ValueError, match="Unsupported document format"):
        module.ingest(str(docs if directory else binary))
    assert stored == []
    assert not (tmp_path / "chunks.json").exists()


def test_invalid_utf8_text_fails_before_indexing(monkeypatch, tmp_path):
    module, stored = _ingestion(monkeypatch, tmp_path, "**/*")
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        module.ingest(str(invalid))
    assert stored == []


def test_empty_glob_fails_before_indexing(monkeypatch, tmp_path):
    module, stored = _ingestion(monkeypatch, tmp_path, "**/*.md")
    with pytest.raises(ValueError, match="No documents matched"):
        module.ingest(str(tmp_path))
    assert stored == []
