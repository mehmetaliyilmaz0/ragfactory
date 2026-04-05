# RAG Research Library - Index

> **Total Research**: ~8,900+ lines across 8 documents
> **Research Date**: April 4-5, 2026
> **Purpose**: Foundation knowledge base for building an automated RAG pipeline builder platform

---

## Research Documents

| # | Document | Lines | Topics Covered |
|---|----------|-------|----------------|
| 1 | [Chunking Strategies](chunking-strategies.md) | 1,295 | Fixed-size, Recursive, Semantic, Late, Contextual (Anthropic), Page-level, Agentic, Document-specific, Proposition-based, NVIDIA/FloTorch/Chroma benchmarks |
| 2 | [Embedding Models](embedding-models.md) | 919 | MTEB leaderboard, OpenAI/Cohere/Voyage/Gemini/Jina APIs, BGE-M3/Qwen3/NV-Embed/Nomic OSS, pricing, Matryoshka, fine-tuning, migration, multilingual, code-specific |
| 3 | [Retrieval Strategies](retrieval-strategies.md) | 895 | Dense (HNSW/IVF/PQ), Sparse (BM25/SPLADE), Hybrid (RRF/weighted), Small-to-Big, Sentence Window, Contextual Retrieval, ColBERT, Ensemble, evaluation metrics, caching |
| 4 | [GraphRAG + RAPTOR](graphrag-raptor-deep-dive.md) | 1,219 | Microsoft GraphRAG (4 query modes, Leiden, cost formulas), RAPTOR (GMM, tree construction), LightRAG, LazyGraphRAG, FastGraphRAG, KG-RAG, HippoRAG, PathRAG |
| 5 | [Pre-Retrieval Optimization](pre-retrieval-optimization-techniques.md) | 1,134 | Query Rewriting (DMQR-RAG), HyDE (math+benchmarks), Query Routing, Query Decomposition, Intent Detection, PRF, technique combinations, production deployment |
| 6 | [Post-Retrieval + Generation](post-retrieval-optimization-and-advanced-generation.md) | 1,348 | Cross-encoders, Cohere Rerank, ColBERT, FlashRank, LLM reranking, LLMLingua, lost-in-the-middle, Self-RAG, CRAG, FLARE, Agentic RAG, RAG prompt engineering |
| 7 | [Evaluation + Platforms + Architecture](RAG_RESEARCH_FINDINGS.md) | 1,006 | RAGAS (8 metrics + formulas), DeepEval, TruLens, ARES, Dify/RAGFlow/LangFlow/Flowise/Haystack/Vectara analysis, config schema, code generation, deployment, monitoring, security |
| 8 | [Vector Databases](vector-databases.md) | 1,136 | ChromaDB, Pinecone (serverless+pod), Qdrant (8,500-12,000 QPS benchmark leader), Weaviate, Milvus (GPU/CAGRA), pgvector+pgvectorscale, FAISS, LanceDB, Elasticsearch, Redis, ANN benchmarks, selection decision tree, cost comparison at scale |

---

## Master Report

| Document | Description |
|----------|-------------|
| [SOTA RAG Research Report](../docs/SOTA_RAG_RESEARCH_REPORT.md) | Executive summary + automation feasibility analysis + UI parameter specs + architecture proposal |

---

## Key Findings Across All Research

### Universal Defaults (Start Here)
- **Chunking**: Recursive splitting, 512 tokens, 50-100 overlap (FloTorch 2026: 69% accuracy)
- **Embedding**: Voyage-3-large (best API value) or BGE-M3 (best OSS with hybrid support)
- **Vector DB**: ChromaDB for prototyping, Qdrant for production
- **Search**: Hybrid (BM25 + dense), RRF fusion, k=60
- **Reranking**: Cohere Rerank 3.5 (API) or FlashRank (CPU)
- **LLM**: Claude Sonnet for generation, temperature 0.05
- **Evaluation**: RAGAS (faithfulness + answer_relevancy + context_precision)

### Top Benchmarks
| Technique | Benchmark | Improvement |
|-----------|-----------|-------------|
| Contextual Retrieval (Anthropic) | Retrieval failure reduction | **67%** (with reranking) |
| Hybrid Search (3-way) | Recall vs single method | **15-30%** |
| Agentic RAG | Complex query accuracy | **34% → 78%** |
| RAPTOR + GPT-4 | QuALITY benchmark | **+20%** |
| GraphRAG | Multi-hop accuracy | **86% vs 32%** |
| Proposition Chunking | Recall@5 | **+17-25%** |
| Query Decomposition | Complex queries | **+44 pts** |

### Cost Reference
| Operation | Cost |
|-----------|------|
| Contextual chunking (Anthropic, cached) | $1.02 / 1M doc tokens |
| GraphRAG indexing (GPT-4o-mini) | $0.06 - $50K+ (corpus-dependent) |
| Voyage-3-large embeddings | $0.06 / 1M tokens |
| Cohere Rerank per search | ~$0.001-0.01 |
| RAGAS evaluation per test case | ~5-7 LLM judge calls |
