# Runtime refresh findings — 2026-10-20

Scope: generated pipeline scaffolds and their declared runtime dependencies. Findings below are from direct reads of the Jinja templates, plus the checked-in dependency matrix. The template graph index is partial for all generated templates; the relevant source lines were read directly. No provider calls were made.

## Fix first: generated project cannot reliably build or run as documented

1. **Docker build order conflicts with declared metadata.** `Dockerfile.j2` copies only `pyproject.toml`, then runs `pip install .`; the generated pyproject declares `readme = "README.md"`, but README is copied only after install. PEP 621 defines a string `readme` as a path relative to the pyproject, and Hatch's wheel `packages` option is for package paths, not arbitrary root modules. Copy metadata files before the install and configure packaging for the generated flat modules (`pipeline.py`, `ingestion.py`, `api.py`, `eval.py`) or remove the unnecessary package-install layer and install dependencies directly. [PyPA metadata spec](https://packaging.python.org/specifications/declaring-project-metadata/), [Hatch build selection](https://hatch.pypa.io/latest/config/build/)

2. **The advertised `.env` setup does not load `.env`.** README tells users to copy `.env.example` to `.env`; generated modules read keys through `os.environ[...]` but don't call a dotenv loader. The generated project can therefore raise `KeyError` despite following its own quick start. Add `python-dotenv` to generated dependencies and call `load_dotenv()` from generated runtime entrypoints (including ingestion); environment values already set by the process remain authoritative by default. Compose `env_file` covers container execution only, not local `python` calls. [Docker Compose env files](https://docs.docker.com/reference/compose-file/services/)

3. **Compose template is syntactically legacy, with stale pinned images.** Top-level `version: "3.9"` is obsolete and Compose v2 ignores it. Remove it. The Weaviate, Milvus, etcd and MinIO image pins are years old; update to currently supported tested versions or omit Compose output for configurations that cannot promise a maintained stack. Don't blindly replace old tags with `latest` for stateful databases. [Docker Compose version field](https://docs.docker.com/reference/compose-file/version-and-name/)

## Runtime configurations currently fail at use time

4. **Contextual and proposition chunking are config-shaped dictionaries, not chunkers.** Their stage builders return dicts. LangChain ingestion calls `.split_documents()` on the result; LlamaIndex sets it as a transformation and later passes it to `VectorStoreIndex.from_documents()`. Both paths fail before indexing. Either implement each as a framework-native transformation that invokes its configured model, or reject those options during generation with an explicit unsupported error. LlamaIndex transformations are callable transform components, not arbitrary settings dicts. [LlamaIndex transformation contract](https://docs.llamaindex.ai/en/v0.10.18/module_guides/loading/ingestion_pipeline/transformations.html)

5. **Configured LlamaIndex answer prompt is dead data.** `pipeline.py.j2` sets `_PROMPT_TEMPLATE` but constructs `RetrieverQueryEngine` without a response synthesizer/prompt override. The config's custom prompt is never used. Pass a `PromptTemplate` to the response synthesizer (`text_qa_template`) when building the query engine. [LlamaIndex prompt override usage](https://docs.llamaindex.ai/en/v0.10.20/module_guides/querying/response_synthesizers/root.html)

6. **Agentic async wrappers can block or fail in active event loops.** The regular LlamaIndex pipeline uses `loop.run_until_complete`; this raises when called while an event loop is already running. The regular FastAPI route is synchronous, which FastAPI runs in a worker thread, but generated callers needing async composition have no async entrypoint for the standard pipeline. Keep a sync convenience wrapper for CLI use, expose a true async API path, and have async web handling await it. [FastAPI async guidance](https://fastapi.tiangolo.com/async/)

## Security and maintenance

7. **Some untrusted config strings are inserted into Python source as quoted text.** LLM templates use `model="{{ llm.model }}"`; LangChain ingestion inserts the configured glob directly into a quoted literal. Quotes, backslashes, or newlines can produce invalid Python or source injection. Use the existing `tojson_py` literal filter for all string-valued config interpolation in Python templates, then retain AST validation.

8. **Dependency upper bounds freeze scaffolds on old major lines.** `versions.py` caps LangChain at `<0.4` and LlamaIndex at `<0.12`, and pins provider integrations to old minor lines. These bounds mean generated projects cannot install newer compatible APIs even when the source templates are updated. Move the matrix as a coordinated set with generated-template compatibility checks; do not simply delete all upper bounds. Provider-specific migration notes should be kept with their model/API templates.

9. **Evaluation needs current APIs and an upstream compatibility pin.** Ragas 0.4 uses `ragas.metrics.collections`, returns `.value`, and expects metric-specific score inputs; AnswerRelevancy also requires an embedding implementation. Ragas 0.4.3 still eagerly imports `ChatVertexAI` from a path removed in `langchain-community` 0.4.2, even when Vertex AI is not used. Keep its evaluation extra on `langchain-community>=0.4,<0.4.2` (the version range overlaps the current LangChain 1.x compatible line), and retain the real SDK smoke test so generated evaluation fails fast on import/API regressions. DeepEval metric constructors accept a configured `model`, and RAG metrics need `retrieval_context`; keep supported metric names aligned with config validation and fail on invalid/non-finite scores. [Ragas 0.3 to 0.4 migration](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/), [Ragas Vertex import issue and compatibility pin](https://github.com/vibrantlabsai/ragas/issues/2745), [DeepEval single-turn evaluation](https://deepeval.com/docs/evaluation-end-to-end-single-turn), [DeepEval contextual precision](https://deepeval.com/docs/metrics-contextual-precision)

## Defer or reject rather than claim support

- LangChain weighted hybrid currently initializes BM25 from an empty corpus, so it is dense-only in practice. A real implementation needs the same ingestion corpus (or persisted sparse index) loaded at query time; otherwise reject weighted hybrid. LlamaIndex's current implementation is query expansion over a dense retriever, not native sparse+dense hybrid.
- Contextual chunking makes one model call per chunk and currently has no batching, retries, caching, or cost control. A minimal implementation can be correct but should not be described as production-ready without those operational controls.
- Compose `depends_on` only controls ordering in the current generated file; it does not mean a database is ready. Add health checks / healthy dependencies only for service images with reliable health probes.
- `reload=True` in the generated Uvicorn `__main__` path is a development setting. Remove it from the default executable path or guard it with an explicit development switch.

## Suggested order

1. Fix generated artifact install/build and `.env` loading; secure Python literal interpolation.
2. Make the configured LlamaIndex prompt effective and provide correct async behavior.
3. Implement or explicitly reject contextual/proposition chunking and weighted hybrid instead of emitting a scaffold that fails or silently degrades.
4. Update dependency bounds and Compose service pins only alongside a repeatable generated-artifact installation/runtime smoke check.

## Sources

- [Python packaging project metadata](https://packaging.python.org/specifications/declaring-project-metadata/)
- [Hatch wheel build selection](https://hatch.pypa.io/latest/config/build/)
- [Docker Compose service `env_file`](https://docs.docker.com/reference/compose-file/services/)
- [Docker Compose obsolete `version`](https://docs.docker.com/reference/compose-file/version-and-name/)
- [LlamaIndex transformations](https://docs.llamaindex.ai/en/v0.10.18/module_guides/loading/ingestion_pipeline/transformations.html)
- [LlamaIndex response synthesizer prompts](https://docs.llamaindex.ai/en/v0.10.20/module_guides/querying/response_synthesizers/root.html)
- [FastAPI async guidance](https://fastapi.tiangolo.com/async/)
- [Ragas 0.3 to 0.4 migration](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/)
- [DeepEval single-turn evaluation](https://deepeval.com/docs/evaluation-end-to-end-single-turn)
- [DeepEval contextual precision](https://deepeval.com/docs/metrics-contextual-precision)
- [Ragas 0.4.3 removed Vertex AI import issue](https://github.com/vibrantlabsai/ragas/issues/2745)
