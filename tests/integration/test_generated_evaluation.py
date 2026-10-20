"""Exercise real Ragas scoring and evaluation constructors without provider calls."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragfactory.core.config import RAGPipelineConfig
from ragfactory.core.generator import generate


@pytest.mark.skipif(
    not os.getenv("RAGFACTORY_EVAL_SMOKE"),
    reason="Run in a generated project's evaluation SDK environment",
)
def test_generated_evaluation_scores_with_compatible_judge_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "smoke-no-network")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    pipeline_stub = ModuleType("pipeline")
    pipeline_stub.run_pipeline = lambda *_args, **_kwargs: ("answer", ["context"])
    monkeypatch.setitem(sys.modules, "pipeline", pipeline_stub)

    config = RAGPipelineConfig.model_validate(
        {
            "name": "eval-smoke",
            "framework": "langchain",
            "indexing": {
                "embedding": {"type": "openai"},
                "vector_db": {"type": "qdrant"},
            },
            "generation": {"llm": {"type": "openai"}},
            "evaluation": {"framework": "both", "metrics": ["faithfulness"]},
        }
    )
    result = generate(config)
    assert result.validation_passed, result.errors
    source = tmp_path / "eval.py"
    source.write_text(result.files["eval.py"], encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generated_eval_smoke", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from openai.resources.chat.completions import AsyncCompletions
    from openai.types.chat import ChatCompletion

    requests: list[dict[str, object]] = []

    async def mock_chat_completion(self, *args, **kwargs):
        requests.append(kwargs)
        payload = (
            {"statements": ["The answer is supported by context."]}
            if len(requests) == 1
            else {
                "statements": [
                    {
                        "statement": "The answer is supported by context.",
                        "reason": "The context supports the answer.",
                        "verdict": 1,
                    }
                ]
            }
        )
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-6-luna",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    monkeypatch.setattr(AsyncCompletions, "create", mock_chat_completion)
    assert module._run_ragas(
        [
            {
                "question": "What does the context support?",
                "answer": "The answer is supported by context.",
                "contexts": ["The context supports the answer."],
                "ground_truth": "The answer is supported by context.",
            }
        ]
    )
    assert "PASS: Metric 'faithfulness' score 1.00" in capsys.readouterr().out
    assert len(requests) == 2
    assert all(request["model"] == "gpt-6-luna" for request in requests)
    assert all(request.get("reasoning_effort") == "none" for request in requests)
    assert all(request.get("temperature") == 1.0 for request in requests)
    module._METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision"]
    assert module._run_ragas([])
    assert module._run_deepeval([])
