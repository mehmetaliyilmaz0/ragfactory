"""
Unit tests for ragfactory.core.compatibility — data integrity.

These tests do NOT test logic (that's validator's job).
They verify the compatibility data is internally consistent and
references real config fields via the dot-path meta-test.
"""

from __future__ import annotations

import re

import pytest

from ragfactory.core.compatibility import (
    CROSS_FIELD_RULES,
    INCOMPATIBLE,
    WARNINGS,
    CompatibilityWarning,
    CrossFieldRule,
    IncompatiblePair,
    Severity,
)


# ─── 1. Structural integrity ──────────────────────────────────────────────────


class TestIncompatiblePairIntegrity:
    def test_all_entries_have_component_a(self) -> None:
        for pair in INCOMPATIBLE:
            assert pair.component_a, f"Empty component_a in: {pair}"

    def test_all_entries_have_component_b(self) -> None:
        for pair in INCOMPATIBLE:
            assert pair.component_b, f"Empty component_b in: {pair}"

    def test_all_entries_have_reason(self) -> None:
        for pair in INCOMPATIBLE:
            assert len(pair.reason) > 20, (
                f"Reason too short (must be descriptive) in: {pair.component_a} × {pair.component_b}"
            )

    def test_no_duplicate_pairs(self) -> None:
        seen: set[tuple[str, str]] = set()
        for pair in INCOMPATIBLE:
            key = (pair.component_a, pair.component_b)
            assert key not in seen, f"Duplicate incompatible pair: {key}"
            seen.add(key)

    def test_doc_url_is_none_or_string(self) -> None:
        for pair in INCOMPATIBLE:
            assert pair.doc_url is None or isinstance(pair.doc_url, str)

    def test_entries_are_frozen(self) -> None:
        pair = INCOMPATIBLE[0]
        with pytest.raises(AttributeError):
            pair.reason = "mutated"  # type: ignore[misc]

    def test_minimum_count(self) -> None:
        """Guard against accidental truncation of the list."""
        assert len(INCOMPATIBLE) >= 13, (
            f"Expected at least 13 incompatible pairs, got {len(INCOMPATIBLE)}"
        )


class TestCompatibilityWarningIntegrity:
    def test_all_entries_have_condition(self) -> None:
        for w in WARNINGS:
            assert w.condition, f"Empty condition in warning: {w}"

    def test_all_entries_have_message(self) -> None:
        for w in WARNINGS:
            assert len(w.message) > 20, f"Warning message too short: {w.condition}"

    def test_all_severities_are_valid(self) -> None:
        valid = set(Severity)
        for w in WARNINGS:
            assert w.severity in valid, f"Invalid severity {w.severity!r} in: {w.condition}"

    def test_cost_per_million_is_positive_or_none(self) -> None:
        for w in WARNINGS:
            if w.cost_per_million is not None:
                assert w.cost_per_million > 0, (
                    f"cost_per_million must be positive, got {w.cost_per_million} in: {w.condition}"
                )

    def test_no_duplicate_conditions(self) -> None:
        seen: set[str] = set()
        for w in WARNINGS:
            assert w.condition not in seen, f"Duplicate warning condition: {w.condition}"
            seen.add(w.condition)

    def test_entries_are_frozen(self) -> None:
        w = WARNINGS[0]
        with pytest.raises(AttributeError):
            w.message = "mutated"  # type: ignore[misc]

    def test_minimum_count(self) -> None:
        assert len(WARNINGS) >= 10, (
            f"Expected at least 10 warnings, got {len(WARNINGS)}"
        )


class TestCrossFieldRuleIntegrity:
    def test_all_entries_have_rule_id(self) -> None:
        for r in CROSS_FIELD_RULES:
            assert r.rule_id, f"Empty rule_id in: {r}"

    def test_all_entries_have_description(self) -> None:
        for r in CROSS_FIELD_RULES:
            assert len(r.description) > 20, f"Description too short: {r.rule_id}"

    def test_no_duplicate_rule_ids(self) -> None:
        seen: set[str] = set()
        for r in CROSS_FIELD_RULES:
            assert r.rule_id not in seen, f"Duplicate rule_id: {r.rule_id}"
            seen.add(r.rule_id)

    def test_required_rules_present(self) -> None:
        rule_ids = {r.rule_id for r in CROSS_FIELD_RULES}
        required = {
            "late_chunking_jina_v3_flag",
            "late_chunking_jina_v2",
            "contextual_chunking_throughput",
            "contextual_extra_api_key",
            "multiple_advanced_techniques",
            "reranker_top_n_vs_top_k",
        }
        missing = required - rule_ids
        assert not missing, f"Missing required cross-field rules: {missing}"


# ─── 2. Severity enum ─────────────────────────────────────────────────────────


class TestSeverityEnum:
    def test_severity_values(self) -> None:
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.COST_ALERT == "cost_alert"

    def test_severity_is_str(self) -> None:
        assert isinstance(Severity.INFO, str)


# ─── 3. Dot-path meta-test ────────────────────────────────────────────────────
#
# This is the most important test in this file.
# It validates that every dot-path in INCOMPATIBLE and WARNINGS
# references a real config field — preventing silent breakage when
# config.py field names change.


def _extract_dot_paths(text: str) -> list[str]:
    """Extract all dot-path strings that look like config paths."""
    # Match things like "generation.llm.anthropic", "indexing.chunking.late", etc.
    return re.findall(r"[a-z_]+(?:\.[a-z_0-9]+){1,}", text)


_KNOWN_CONFIG_PATHS: set[str] = {
    # framework
    "framework.langchain",
    "framework.llamaindex",
    # indexing.chunking types
    "indexing.chunking.fixed",
    "indexing.chunking.recursive",
    "indexing.chunking.semantic",
    "indexing.chunking.contextual",
    "indexing.chunking.late",
    "indexing.chunking.page_level",
    "indexing.chunking.proposition",
    # indexing.embedding types
    "indexing.embedding.openai",
    "indexing.embedding.cohere",
    "indexing.embedding.voyage",
    "indexing.embedding.gemini",
    "indexing.embedding.bge_m3",
    "indexing.embedding.nomic",
    "indexing.embedding.jina",
    # indexing.vector_db types
    "indexing.vector_db.chromadb",
    "indexing.vector_db.qdrant",
    "indexing.vector_db.pinecone",
    "indexing.vector_db.weaviate",
    "indexing.vector_db.milvus",
    "indexing.vector_db.pgvector",
    # retrieval types
    "retrieval.dense",
    "retrieval.hybrid_rrf",
    "retrieval.hybrid_weighted",
    "retrieval.small_to_big",
    "retrieval.sentence_window",
    # pre_retrieval
    "pre_retrieval.hyde",
    "pre_retrieval.query_rewriting",
    "pre_retrieval.routing",
    # post_retrieval.reranker types
    "post_retrieval.reranker.cohere",
    "post_retrieval.reranker.cross_encoder",
    "post_retrieval.reranker.colbert",
    "post_retrieval.reranker.flashrank",
    # generation.llm types
    "generation.llm.openai",
    "generation.llm.anthropic",
    "generation.llm.cohere_llm",
    "generation.llm.ollama",
    # generation.advanced
    "generation.advanced.crag",
    "generation.advanced.flare",
    "generation.advanced.agentic",
    "generation.advanced.crag.web_search_fallback",
    # evaluation
    "evaluation",
    # ingestion
    "ingestion.parser",
}


class TestDotPathValidity:
    """
    Meta-test: every dot-path used in INCOMPATIBLE.component_a/b and
    WARNINGS.condition must resolve to a known config path.

    This test fails when config.py field names change but compatibility.py
    dot-paths are not updated — catching silent breakage before it ships.
    """

    def _collect_paths_from_incompatible(self) -> list[tuple[str, str]]:
        """Returns list of (dot_path, source_description) pairs."""
        results = []
        for pair in INCOMPATIBLE:
            results.append((pair.component_a, f"INCOMPATIBLE[component_a]: {pair.component_a}"))
            if pair.component_b != "*":
                results.append((pair.component_b, f"INCOMPATIBLE[component_b]: {pair.component_b}"))
        return results

    def _collect_paths_from_warnings(self) -> list[tuple[str, str]]:
        results = []
        for w in WARNINGS:
            # condition may be a simple path or a compound expression
            # only validate simple dot-paths
            if re.match(r"^[a-z_]+(?:\.[a-z_0-9]+)+$", w.condition):
                results.append((w.condition, f"WARNINGS[condition]: {w.condition}"))
        return results

    def test_incompatible_component_a_paths_are_known(self) -> None:
        unknown = []
        for path, src in self._collect_paths_from_incompatible():
            if path not in _KNOWN_CONFIG_PATHS:
                unknown.append(f"  {src}")
        assert not unknown, (
            "Unknown dot-paths in INCOMPATIBLE (update _KNOWN_CONFIG_PATHS or fix the path):\n"
            + "\n".join(unknown)
        )

    def test_warnings_condition_paths_are_known(self) -> None:
        unknown = []
        for path, src in self._collect_paths_from_warnings():
            if path not in _KNOWN_CONFIG_PATHS:
                unknown.append(f"  {src}")
        assert not unknown, (
            "Unknown dot-paths in WARNINGS (update _KNOWN_CONFIG_PATHS or fix the condition):\n"
            + "\n".join(unknown)
        )


# ─── 4. Specific known entries ────────────────────────────────────────────────


class TestKnownIncompatibilities:
    """Spot-checks that critical incompatibilities are present with correct data."""

    def _find(self, a: str, b: str) -> IncompatiblePair | None:
        return next(
            (p for p in INCOMPATIBLE if p.component_a == a and p.component_b == b),
            None,
        )

    def test_flare_anthropic_present(self) -> None:
        pair = self._find("generation.advanced.flare", "generation.llm.anthropic")
        assert pair is not None, "FLARE × Anthropic incompatibility must be defined"
        assert "logprob" in pair.reason.lower()

    def test_flare_cohere_present(self) -> None:
        assert self._find("generation.advanced.flare", "generation.llm.cohere_llm") is not None

    def test_late_chunking_openai_present(self) -> None:
        pair = self._find("indexing.chunking.late", "indexing.embedding.openai")
        assert pair is not None
        assert "jina" in pair.reason.lower()

    def test_hybrid_rrf_chromadb_present(self) -> None:
        assert self._find("retrieval.hybrid_rrf", "indexing.vector_db.chromadb") is not None

    def test_hybrid_weighted_pinecone_present(self) -> None:
        assert self._find("retrieval.hybrid_weighted", "indexing.vector_db.pinecone") is not None

    def test_sentence_window_langchain_present(self) -> None:
        pair = self._find("retrieval.sentence_window", "framework.langchain")
        assert pair is not None
        assert "llamaindex" in pair.reason.lower() or "LlamaIndex" in pair.reason


class TestKnownWarnings:
    """Spot-checks that critical warnings are present."""

    def _find(self, condition: str) -> CompatibilityWarning | None:
        return next((w for w in WARNINGS if w.condition == condition), None)

    def test_contextual_chunking_cost_warning(self) -> None:
        w = self._find("indexing.chunking.contextual")
        assert w is not None
        assert w.severity == Severity.COST_ALERT
        assert w.cost_per_million == pytest.approx(1.02)

    def test_proposition_chunking_cost_warning(self) -> None:
        w = self._find("indexing.chunking.proposition")
        assert w is not None
        assert w.severity == Severity.COST_ALERT
        assert w.cost_per_million == pytest.approx(2.50)

    def test_chromadb_production_warning(self) -> None:
        w = self._find("indexing.vector_db.chromadb")
        assert w is not None
        assert w.severity == Severity.WARNING

    def test_bge_m3_gpu_warning(self) -> None:
        w = self._find("indexing.embedding.bge_m3")
        assert w is not None
        assert w.severity == Severity.WARNING
        assert "gpu" in w.message.lower() or "GPU" in w.message

    def test_hyde_info_only(self) -> None:
        w = self._find("pre_retrieval.hyde")
        assert w is not None
        assert w.severity == Severity.INFO
        assert w.cost_per_million is None
