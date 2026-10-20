# Modernization verification, 2026-10-20

Release candidate: 0.2.0. This verification was recorded before commit and push.
No package publication or deployment was performed. Model, SDK, and runtime
research is linked from the root README.

## Changes and compatibility

Generated projects target LangChain 1.x / LangGraph 1.x or LlamaIndex core 0.14.
Provider adapters retain independently versioned constraints. Generated code still
has no runtime import of `ragfactory`.

The default OpenAI chat model is `gpt-5.4-mini`, verified against the installed
LlamaIndex model registry. Anthropic defaults to `claude-sonnet-4-6` for the same
SDK-compatibility reason. Model IDs are no longer frozen to a schema allowlist.
Embedding defaults and dimension mappings were refreshed, and retired Google
embedding IDs now fail with migration guidance.

Changing an embedding model or its dimension requires a new collection and full
reindexing. Existing contextual, proposition, and late chunking configurations are
parseable but cannot generate their incomplete runtime implementations, including
with `--force`. LlamaIndex hybrid retrieval is limited to Qdrant. These are
intentional compatibility changes, not newly implemented advanced features.

## Verification evidence

- Python 3.14.6: full suite, 637 passed, four opt-in SDK cases skipped.
- Python 3.12.15: full suite, 637 passed, four opt-in SDK cases skipped.
- Ruff lint and formatting checks pass. Strict Mypy passes for 11 source files.
- Separate real-SDK smoke tests passed for LangChain and LlamaIndex: generated
  text ingestion, local Qdrant indexing, retrieval, and fake-model generation.
  The LlamaIndex smoke also checks the custom prompt and model metadata registry.
- Separate evaluation smoke passed against real Ragas 0.4.3 and DeepEval 4.3.1.
  It runs Ragas Faithfulness scoring through intercepted OpenAI calls and checks
  default judge request parameters. All supported Ragas and DeepEval metrics are
  constructed; other metric scoring and DeepEval `measure` are not exercised.
- Installed dense SDK profiles pass `uv pip check`. Representative Voyage /
  Anthropic / Cohere hybrid, Google GenAI / Anthropic, and local HuggingFace
  dependency profiles were resolver-checked without installing hardware stacks.
- Isolated wheel and sdist builds passed, as did Twine metadata checks. The wheel
  contains 46 template assets, including the dot-prefixed environment template.
  The source archive excludes virtual environments and real `.env` credentials.
- CI now runs the full suite and static checks on Python 3.11 through 3.14,
  plus real generated-runtime jobs on Python 3.12. Configuring CI is not evidence
  of a successful hosted run; the local runs above are the executed evidence.

## Self-Feedback & Resolution Log

### Principal software architect

Configuration validation, package selection, and rendering remain separate.
Incomplete features are rejected by shared validation used both by the validator
and generator, preventing CLI `--force` from emitting broken implementations.
LangChain legacy retrievers use `langchain-classic`. LlamaIndex retrieval uses
async APIs and its configured response prompt. No new framework abstraction was
introduced.

### QA and security engineer

- Found unsafe interpolation of vector-store and reranker strings. Switched to
  Python literal rendering and added hostile quote/newline regression cases.
- Found a semantic splitter hardcoded to OpenAI and an enum/string rendering
  mismatch. It now uses the configured embedding adapter; explicit threshold
  configurations are covered by regression tests.
- Found an empty BM25 corpus and incorrect LlamaIndex hybrid behavior. LangChain
  uses ingestion's persisted chunks; Qdrant has sparse indexing and an explicit
  reciprocal-rank callback with behavioral tests.
- Found that raw SentenceTransformer / FlagEmbedding objects do not implement
  LangChain's adapter interfaces. Generated code now uses the framework wrappers.
- Found the old LlamaIndex Gemini package range had no published candidate.
  Resolver evidence led to `llama-index-embeddings-google-genai` and its matching
  import and constructor.
- Evaluation tests cover malformed datasets, missing references, metric-specific
  arguments, and non-finite scores. Real SDK import verification caught a Ragas
  dependency problem that the executable stubs could not catch. Ragas 0.4.3 imports
  a module removed in langchain-community 0.4.2; evaluation profiles constrain
  community to `>=0.4,<0.4.2`. Import and constructor smoke passed with 0.4.1.

### Release manager

- Fixed Docker's install-before-copy sequence and generated flat-module wheel
  configuration. Standalone generated projects installed in isolated SDK venvs.
- A real sdist build included local `.venv-*` directories and failed extraction
  on absolute Python symlinks. Added ignore rules and an explicit source archive
  include list; the subsequent build and archive-content checks passed.
- Bumped package metadata and runtime version together to 0.2.0. No release was
  published and the user's existing indexes or generated projects were not edited.

## Limits and operational risks

No paid provider calls, model downloads, GPU inference, Docker daemon builds,
hosted CI runs, or external vector database services were tested. Python 3.11 and
3.13 are configured in CI but were not exercised locally. Only representative
dependency profiles were resolved, not every provider/store combination.

Compose remains development scaffolding, including legacy optional database
images and development credentials. Parent-document / sentence-window metadata,
alternate ingestion sources/parsers, routing, and compression are not all
implemented. LangChain community/experimental packages have maintenance risks.
Repeated ingestion appends chunks to both retrieval storage and the BM25 sidecar;
this update does not add idempotent ingestion or cross-store transactions.
Back up generated projects before regenerating into the same directory.

## Independent review follow-up

The independent Luna review found an incompatible default Ragas request,
silent narrowing of an explicit ingestion glob, and ignored BM25 parameters.
The fixes are checked with regression tests rather than constructor-only smoke:

- Explicit `**/*` ingestion now includes Markdown and text together. Unsupported
  extensions and malformed UTF-8 fail before any documents are indexed. Empty
  globs fail explicitly. Alternative document parsers are still not implemented.
- Both LangChain hybrid strategies pass `bm25_k1` and `bm25_b` into BM25. Real
  SDK tests observe non-default values in the actual rank-bm25 vectorizer and run
  retrieval against local Qdrant.
- The real-SDK runtime test now exercises dense, RRF hybrid, and weighted hybrid
  for both frameworks. LlamaIndex uses a deterministic sparse test encoder instead
  of downloading a model; its Qdrant sparse indexing, querying, and fusion stay
  real. These tests also exposed and fixed the LlamaIndex `VectorStoreQueryMode`
  import, which must come from `llama_index.core.vector_stores.types`.
- Default Ragas Luna judge requests now explicitly set `reasoning_effort="none"`,
  compatible with the temperature value injected by Ragas 0.4.3. Other configured
  judge models retain their original settings. The evaluation regression runs
  actual Ragas Faithfulness scoring through Instructor with the OpenAI request
  boundary intercepted. It checks both generated requests and a final score of
  1.0, then constructs all three supported metrics in both evaluation frameworks.
- Independent Luna re-review cleared the glob and BM25 fixes after rerunning the
  five input regressions and all six framework/retrieval SDK smoke cases.
  A separate bounded P1 re-review reran the real Ragas scoring regression and
  cleared the default judge fix. No live provider acceptance was claimed.
