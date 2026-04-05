# State-of-the-Art RAG Paradigms: Deep Research Report for Automated RAG Builder Platform

> **Date**: April 4, 2026
> **Purpose**: Research foundation for building a UI-driven, automated RAG pipeline builder platform
> **Scope**: SOTA RAG paradigms, automation feasibility analysis, and system architecture recommendation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SOTA RAG Paradigms & Automation Analysis](#2-sota-rag-paradigms--automation-analysis)
   - 2.1 [RAG Taxonomy: Naive vs Advanced vs Modular](#21-rag-taxonomy-naive-vs-advanced-vs-modular)
   - 2.2 [Pre-Retrieval Optimization](#22-pre-retrieval-optimization)
   - 2.3 [Retrieval Strategies](#23-retrieval-strategies)
   - 2.4 [Post-Retrieval Optimization](#24-post-retrieval-optimization)
   - 2.5 [Generation & Evaluation](#25-generation--evaluation)
3. [Recommended Pipeline Modules](#3-recommended-pipeline-modules)
4. [System Architecture Proposal](#4-system-architecture-proposal)
5. [Blind Spots & Edge Cases](#5-blind-spots--edge-cases)

---

## 1. Executive Summary

The RAG landscape in 2025-2026 has matured from simple "retrieve-and-read" pipelines into a **Modular RAG** paradigm where each stage (indexing, pre-retrieval, retrieval, post-retrieval, generation, orchestration) is an independent, composable module with well-defined parameters (arXiv:2407.21059, July 2024).

**Key findings from this research:**

- **Standard RAG achieves only ~63% accuracy** on complex benchmarks (Meta CRAG Benchmark). Advanced techniques like Agentic RAG push this to **78% on complex queries** — a 44-point improvement.
- **Hybrid search (dense + sparse + reranking)** is now the consensus production baseline, delivering **15-30% better recall** than either method alone.
- **Anthropic's Contextual Retrieval** reduces retrieval failures by **49-67%** when combining contextual embeddings + contextual BM25 + reranking.
- **Every major RAG technique can be parameterized** into UI configuration. The automation feasibility ranges from High (chunking, embedding selection, hybrid search) to Medium-Low (Self-RAG requiring model fine-tuning, GraphRAG requiring domain-specific tuning).
- **Configuration-driven RAG** is a proven pattern. Pathway, customrag, and UltraRAG already demonstrate YAML-based pipeline definition. Our platform should adopt this as the intermediate representation between UI and code generation.
- **The existing tooling ecosystem** (Dify at 90.5k stars, RAGFlow at 48.5k, LangChain at 105k) validates market demand but focuses on visual workflow builders, not on generating deployable, optimized pipeline code — which is our differentiator.

**Recommended architecture**: A **configuration-first** approach where the UI produces a structured YAML/JSON spec, a **Code Generation Engine** translates this into LangChain/LlamaIndex code, and a **Deployment Layer** packages it as a containerized microservice. This avoids framework lock-in while providing deterministic, auditable output.

---

## 2. SOTA RAG Paradigms & Automation Analysis

### 2.1 RAG Taxonomy: Naive vs Advanced vs Modular

#### Concept & Value

RAG systems exist in an evolutionary hierarchy:

| Paradigm | Architecture | Strengths | Limitations |
|----------|-------------|-----------|-------------|
| **Naive RAG** | Linear: Index → Retrieve → Generate | Simple, fast to implement | Poor precision/recall, hallucination, can't handle complex queries |
| **Advanced RAG** | Linear + pre/post-retrieval optimization | Better quality, still manageable | Rigid linear flow, single retrieval strategy |
| **Modular RAG** | Composable modules with routing, scheduling, fusion | Maximum flexibility, domain-adaptable | Higher complexity, requires orchestration |

Modular RAG (arXiv:2407.21059) defines **six modules**: Indexing, Pre-retrieval, Retrieval, Post-retrieval, Generation, Orchestration. Each module contains **operators** (e.g., Indexing has: chunk optimization, metadata enrichment, hierarchical indexing, knowledge graph construction). Modules compose via **five flow patterns**: Linear, Conditional, Branching, Looping, and Tuning.

**This is the foundational model for our platform.** Each module maps to a UI section. Each operator maps to a toggle/configuration panel.

#### Automation Feasibility: **HIGH**

The modular taxonomy is inherently designed for composition. The UI can present RAG "tiers" (Naive/Advanced/Modular) as complexity presets, with individual modules exposed as configurable sections.

#### Configuration Parameters (UI)

| Parameter | Type | Values |
|-----------|------|--------|
| `pipeline_tier` | Preset selector | `naive`, `advanced`, `modular` |
| `flow_pattern` | Selector | `linear`, `conditional`, `branching`, `looping` |
| `enabled_modules` | Checklist | `[indexing, pre_retrieval, retrieval, post_retrieval, generation, orchestration]` |

---

### 2.2 Pre-Retrieval Optimization

#### 2.2.1 Query Rewriting / Multi-Query Expansion

**Concept & Value**: Generate multiple rephrasings of a query to improve recall. A single query may miss relevant documents due to vocabulary mismatch. Multi-query expansion generates 3-5 variants and retrieves across all of them, then deduplicates results. Performance impact: Complex query accuracy jumps from **~34% to ~78%** when combined with routing.

**When best used**: Ambiguous queries, short queries, when users don't know exact terminology.

**Automation Feasibility: HIGH**
This is a straightforward LLM call with a system prompt. Fully parameterizable — the developer only needs to enable it and set the number of rewrites.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable query rewriting |
| `strategy` | Enum | `multi_query` | `multi_query`, `sub_question`, `step_back` |
| `num_rewrites` | Integer | `3` | Number of query variants to generate |
| `rewrite_model` | Selector | (inherit from LLM) | Model for rewriting (can use cheaper model) |
| `aggregation` | Enum | `merge_deduplicate` | How to combine results: `merge_deduplicate`, `reciprocal_rank_fusion` |

---

#### 2.2.2 HyDE (Hypothetical Document Embeddings)

**Concept & Value**: Instead of embedding the raw query, use an LLM to generate a hypothetical answer document, then embed *that* for retrieval. This converts query-to-document matching into document-to-document matching, which aligns better with how embedding models are trained (arXiv:2212.10496).

**When best used**: Short/ambiguous queries, vocabulary gap between user language and document language, exploratory search.
**When to avoid**: Exact keyword lookups, latency-critical paths, domains the LLM doesn't know well.

**Automation Feasibility: HIGH**
Single LLM call with configurable prompt. No model fine-tuning needed. Can be toggled on/off as a pre-retrieval step.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable HyDE |
| `hyde_model` | Selector | (inherit from LLM) | Model for hypothesis generation |
| `num_hypotheses` | Integer | `1` | Generate multiple hypotheses and average embeddings |
| `hyde_prompt_template` | Text | (built-in) | Custom prompt for generating hypothetical documents |
| `temperature` | Float | `0.7` | Creativity of hypothesis generation |

---

#### 2.2.3 Query Routing

**Concept & Value**: Classify incoming queries and dispatch them to the most appropriate retrieval strategy or data source. Acts as an intelligent dispatcher — e.g., factoid queries use keyword search, analytical queries use multi-hop retrieval, code queries hit a code-specific index.

**When best used**: Multiple data sources or retrieval strategies available, queries vary significantly in type.

**Automation Feasibility: MEDIUM**
Routing logic requires domain-specific knowledge about available indices and their contents. The UI can expose route definitions, but the developer needs to specify the routing rules.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable query routing |
| `routes` | Array | `[]` | List of `{name, description, retriever_config}` |
| `routing_model` | Selector | (inherit from LLM) | Model for classification |
| `confidence_threshold` | Float | `0.7` | Minimum confidence for routing |
| `fallback_route` | Selector | `default` | Route when no confident match |

---

### 2.3 Retrieval Strategies

#### 2.3.1 Chunking Strategies

**Concept & Value**: How documents are split into retrieval units. This is the **single most impactful decision** in a RAG pipeline — bad chunks make everything downstream worse.

| Strategy | Recall Benchmark | Best For | Complexity |
|----------|-----------------|----------|------------|
| **Fixed-size** | 85-89% | Prototyping, homogeneous docs | Trivial |
| **Recursive character** | 85-90% | General-purpose, structured text | Low |
| **Semantic chunking** | ~91.9% | Long docs with topic shifts | Medium |
| **Late chunking** | High efficiency | Context-dependent text | Medium |
| **Contextual chunking** (Anthropic) | 49-67% failure reduction | High-stakes, ambiguous chunks | Medium-High |
| **Page-level** | 64.8% accuracy (NVIDIA best) | PDFs, formatted documents | Low |

**Key insight from NVIDIA 2024 benchmark**: Page-level chunking won overall accuracy (0.648). Factoid queries work best with 256-512 tokens; analytical queries with 1024+ tokens. **Chunk size should be adaptive to query type.**

**Automation Feasibility: HIGH**
All parameters are numeric or enum. The UI can provide presets ("Conservative", "Balanced", "Aggressive") plus manual overrides.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | Enum | `recursive` | `fixed`, `recursive`, `semantic`, `late`, `contextual`, `page_level` |
| `chunk_size` | Integer | `512` | Tokens per chunk |
| `chunk_overlap` | Integer | `50` | Overlap between chunks |
| `separators` | Array | `["\n\n", "\n", " "]` | For recursive splitting |
| `semantic_threshold` | Float | `0.85` | Similarity breakpoint for semantic chunking |
| `context_model` | Selector | `claude-haiku` | Model for contextual chunking |
| `context_prompt` | Text | (Anthropic default) | Custom contextual prompt |

---

#### 2.3.2 Dense / Sparse / Hybrid Search

**Concept & Value**: Production RAG should almost always use **hybrid search** — running BM25 (sparse/keyword) and dense vector retrieval in parallel, then fusing results. IBM research confirms **three-way hybrid** (BM25 + dense + SPLADE) is optimal. Hybrid delivers **15-30% better recall** than either method alone.

**Fusion strategies**:
- **Reciprocal Rank Fusion (RRF)**: `score = Σ 1/(k + rank_i)`, k=60. Simple, effective, no tuning needed.
- **Weighted Fusion**: `score = α × dense + (1-α) × sparse`. More control, requires alpha tuning.

**Automation Feasibility: HIGH**
Hybrid search is essentially a toggle + weight. All vector DBs now support it natively.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_type` | Enum | `hybrid` | `dense`, `sparse`, `hybrid` |
| `fusion_strategy` | Enum | `rrf` | `rrf`, `weighted` |
| `alpha` | Float | `0.5` | Dense vs sparse weight (weighted fusion) |
| `rrf_k` | Integer | `60` | RRF constant |
| `top_k` | Integer | `20` | Candidates per retriever |
| `bm25_k1` | Float | `1.2` | BM25 term saturation |
| `bm25_b` | Float | `0.75` | BM25 length normalization |
| `use_splade` | Boolean | `false` | Enable learned sparse vectors |

---

#### 2.3.3 Small-to-Big Retrieval

**Concept & Value**: Decouple the **retrieval unit** (small chunks for precision) from the **synthesis unit** (large parent chunks for context). Index small child chunks (128-256 tokens) that reference larger parent chunks (512-2048 tokens). Retrieve against small chunks, but pass parent chunks to the LLM. This solves the "embedding dilution" problem where large chunks contain too much irrelevant filler.

**Automation Feasibility: HIGH**
LlamaIndex provides `ParentDocumentRetriever` out of the box. Configuration is purely numeric.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable small-to-big |
| `child_chunk_size` | Integer | `256` | Small chunk for retrieval |
| `parent_chunk_size` | Integer | `1024` | Large chunk for synthesis |
| `child_overlap` | Integer | `25` | Overlap for child chunks |
| `top_k` | Integer | `5` | Number of parent chunks to retrieve |

---

#### 2.3.4 Sentence Window Retrieval

**Concept & Value**: Retrieve individual sentences for maximum precision, then expand context by including surrounding sentences. LlamaIndex's `SentenceWindowNodeParser` stores a window of ±N sentences as metadata. At synthesis time, `MetadataReplacementNodePostProcessor` replaces each matched sentence with its full window.

**Automation Feasibility: HIGH**
Single numeric parameter (window size) with LlamaIndex native support.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable sentence window |
| `window_size` | Integer | `5` | Sentences on each side |
| `top_k` | Integer | `5` | Number of sentence matches |

---

#### 2.3.5 GraphRAG

**Concept & Value**: Microsoft's GraphRAG (arXiv:2404.16130) builds a knowledge graph from the corpus using LLM-extracted entities/relationships, applies Leiden community detection for hierarchical clustering, and generates community summaries at each level. Excels at **global sensemaking** questions ("What are the main themes?") that standard vector search fundamentally cannot answer.

**Four query modes**: Global (corpus-wide themes via community summaries), Local (entity-neighborhood exploration), DRIFT (dynamic community search), Basic (standard vector fallback).

**When best used**: Large corpora requiring holistic understanding, entity-relationship analysis, narrative data.
**When to avoid**: Simple factoid Q&A, small corpora, frequently changing data, real-time latency requirements.

**Automation Feasibility: MEDIUM**
GraphRAG indexing is compute-intensive (many LLM calls for entity extraction) and requires domain-aware tuning of community detection parameters. Can be offered as an "advanced" option with sensible defaults, but the developer needs to understand the trade-offs (cost, latency, indexing time).

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable GraphRAG |
| `query_mode` | Enum | `local` | `global`, `local`, `drift`, `basic` |
| `max_cluster_size` | Integer | `10` | Max entities per leaf community |
| `community_level` | Integer | `2` | Leiden hierarchy level |
| `entity_extraction_model` | Selector | (inherit from LLM) | LLM for entity extraction |
| `allow_general_knowledge` | Boolean | `false` | Include real-world knowledge |
| `drift_k_followups` | Integer | `5` | Top results for DRIFT search |

---

#### 2.3.6 RAPTOR

**Concept & Value**: Recursively embed, cluster, and summarize text chunks into a tree structure (ICLR 2024, Stanford). Retrieval can happen from any tree level — leaf nodes have details, upper nodes have abstractions. Achieved **20% improvement on QuALITY benchmark** with GPT-4. Best for book-length documents requiring multi-hop reasoning across different granularity levels.

**Automation Feasibility: MEDIUM**
Tree construction is expensive (many LLM summarization calls). Configuration involves clustering algorithm selection and tree depth — can be automated with good defaults but requires user awareness of cost implications.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable RAPTOR |
| `clustering_algorithm` | Enum | `gmm` | `gmm`, `kmeans` |
| `num_clusters` | Integer | `10` | Clusters per level |
| `tree_depth` | Integer | `3` | Max recursive levels |
| `summarization_model` | Selector | (inherit from LLM) | LLM for cluster summarization |
| `retrieval_mode` | Enum | `collapsed` | `tree_traversal`, `collapsed` |

---

### 2.4 Post-Retrieval Optimization

#### 2.4.1 Re-ranking Models

**Concept & Value**: Second-stage filter that reorders initial retrieval results by true relevance. Cross-encoders process query-document pairs jointly for deep understanding. **Almost always recommended** in production — retrieve 50-150 candidates, rerank to top 5-20.

| Model Type | Accuracy | Speed | Cost |
|-----------|----------|-------|------|
| **Cross-Encoder** (ms-marco-MiniLM) | Highest | Slowest | GPU required |
| **ColBERT** (late interaction) | High | Fast (precomputed docs) | Medium |
| **Cohere Rerank 3.5** | High | Fast (API) | Per-query pricing |
| **FlashRank** | Good | Fastest | CPU-friendly |
| **LLM-based** (RankGPT) | Very High | Very Slow | Very High |

**Automation Feasibility: HIGH**
Model selection + top_n is the entire configuration. All major libraries support pluggable rerankers.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `true` | Enable reranking |
| `model` | Selector | `cohere-rerank-3.5` | Reranker model |
| `top_n` | Integer | `5` | Final documents after reranking |
| `initial_top_k` | Integer | `100` | Candidates to rerank |
| `score_threshold` | Float | `0.0` | Minimum relevance cutoff |

---

#### 2.4.2 Context Compression

**Concept & Value**: Reduce retrieved context to only the most relevant portions before passing to the LLM. Addresses the **"lost-in-the-middle"** effect where LLMs degrade >30% when relevant information falls in the middle of long contexts. Methods include LLMLingua (token-level compression), extractive summarization, and sentence-level filtering.

**Automation Feasibility: MEDIUM-HIGH**
Can be toggled with a compression ratio, but optimal compression varies by domain and query type.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable context compression |
| `method` | Enum | `extractive` | `llm_lingua`, `extractive`, `sentence_filter` |
| `compression_ratio` | Float | `0.5` | Target compression ratio |
| `max_context_tokens` | Integer | `4096` | Maximum tokens in final context |

---

#### 2.4.3 Prompt Formatting & Context Assembly

**Concept & Value**: How retrieved documents are assembled into the final prompt. Position matters — highest-confidence results should come first (addresses lost-in-the-middle). Format matters — structured templates with clear source delineation improve faithfulness.

**Automation Feasibility: HIGH**
Template-based, fully configurable via prompt editor.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_template` | Text | (built-in) | System prompt template with `{context}` and `{query}` placeholders |
| `context_ordering` | Enum | `relevance_first` | `relevance_first`, `chronological`, `reverse_relevance` |
| `source_attribution` | Boolean | `true` | Include source references |
| `context_separator` | String | `"\n---\n"` | Separator between documents |
| `max_sources` | Integer | `5` | Maximum documents in context |

---

### 2.5 Generation & Evaluation

#### 2.5.1 Self-RAG

**Concept & Value**: Trains the LLM itself to generate **reflection tokens** that control retrieval decisions (ICLR 2024 Oral, top 1%). Three tokens: ISREL (relevance judgment), ISSUP (support verification), ISUSE (usefulness rating). The model dynamically decides whether to retrieve, critiques retrieved passages, and self-corrects.

**Key result**: Self-CRAG variant achieves 61.8% accuracy on PopQA vs 54.9% baseline.

**Automation Feasibility: LOW**
Requires fine-tuning the LLM on reflection token generation. Cannot be achieved through prompting alone. Could be offered as a pre-trained model selection (e.g., Self-RAG-LLaMA-7B), but custom training is out of scope for a UI builder.

**Configuration Parameters (when using pre-trained Self-RAG model):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable Self-RAG |
| `model` | Selector | `self-rag-llama-7b` | Pre-trained Self-RAG model |
| `w_rel` | Float | `1.0` | Weight for relevance judgment |
| `w_sup` | Float | `1.0` | Weight for support verification |
| `w_use` | Float | `0.5` | Weight for usefulness rating |

---

#### 2.5.2 CRAG (Corrective RAG)

**Concept & Value**: Uses a lightweight retrieval evaluator (T5-large, 0.77B params) to assess retrieved document quality and trigger corrective actions (arXiv:2401.15884). Three paths: **Correct** (refine via decompose-then-recompose), **Incorrect** (fallback to web search), **Ambiguous** (combine both). The evaluator achieves 84.3% accuracy vs ChatGPT's 58-64.7%.

**Automation Feasibility: MEDIUM-HIGH**
Can be implemented as a post-retrieval verification step using LLM-as-judge without fine-tuning. Confidence thresholds are dataset-specific but can default to reasonable values.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable CRAG |
| `evaluator_model` | Selector | (inherit from LLM) | Model for retrieval evaluation |
| `upper_threshold` | Float | `0.7` | Confidence above = Correct |
| `lower_threshold` | Float | `-0.5` | Confidence below = Incorrect |
| `web_search_fallback` | Boolean | `true` | Enable web search for Incorrect |
| `web_search_provider` | Selector | `tavily` | Web search API |

---

#### 2.5.3 FLARE (Forward-Looking Active Retrieval)

**Concept & Value**: During generation, monitors token-level confidence. When any token probability drops below a threshold, the generated sentence becomes a retrieval query, new documents are fetched, and generation restarts from that point (EMNLP 2023). Provides continuous fact-checking during generation.

**Automation Feasibility: MEDIUM**
Requires access to token-level probabilities (not available from all LLM APIs). Works well with open-source models but limited with proprietary APIs.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable FLARE |
| `confidence_threshold` | Float | `0.5` | Token probability trigger |
| `max_iterations` | Integer | `5` | Maximum retrieval-regeneration cycles |
| `generation_model` | Selector | — | Must support logprobs output |

---

#### 2.5.4 Agentic RAG

**Concept & Value**: Embeds autonomous AI agents into the RAG pipeline using four design patterns: reflection, planning, tool use, and multi-agent collaboration (arXiv:2501.09136, Jan 2025). Transforms static retrieve-then-generate into adaptive, iterative reasoning. Complex query accuracy: **34% → 78%**.

**Automation Feasibility: MEDIUM**
The routing/reflection loop can be configured via LangGraph or LlamaIndex agent definitions. However, tool definitions and agent behaviors require domain-specific customization.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable agentic mode |
| `agent_framework` | Enum | `langgraph` | `langgraph`, `llamaindex_agents`, `crewai` |
| `max_reasoning_steps` | Integer | `5` | Maximum agent iterations |
| `tools` | Array | `[]` | Available tools: `web_search`, `calculator`, `code_exec` |
| `reflection_enabled` | Boolean | `true` | Agent evaluates its own retrieval quality |

---

#### 2.5.5 RAG Evaluation

**Concept & Value**: Automated evaluation of RAG pipeline quality using LLM-as-judge frameworks. RAGAS (EACL 2024) provides four core metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall. DeepEval adds debuggable scoring with CI/CD integration.

**Target production metrics** (from industry benchmarks):
- Precision@K ≥ 0.75-0.85
- Answer Rate ≥ 0.90
- Mean Time to Answer < 3s
- Hallucination Rate < 10%

**Automation Feasibility: HIGH**
Evaluation can be automatically configured as a pipeline testing stage. All metrics are LLM-computable.

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_framework` | Enum | `ragas` | `ragas`, `deepeval`, `both` |
| `metrics` | Checklist | `[faithfulness, answer_relevancy, context_precision, context_recall]` | Metrics to evaluate |
| `judge_model` | Selector | `gpt-4o` | LLM for evaluation |
| `num_test_cases` | Integer | `50` | Evaluation dataset size |
| `pass_threshold` | Float | `0.7` | Minimum score to pass |

---

## 3. Recommended Pipeline Modules

Based on the research, the platform should expose **seven logical modules** in the UI, mapping directly to the Modular RAG framework:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline Builder UI                      │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
│  Module 1│  Module 2│  Module 3│  Module 4│  Module 5│   Module 6  │
│  DATA    │  INDEXING│  PRE-    │ RETRIEVAL│  POST-   │  GENERATION │
│  INGEST  │          │ RETRIEVAL│          │ RETRIEVAL│             │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────┤
│• Sources │• Chunking│• Query   │• Search  │• Reranker│• LLM Model  │
│• Parsers │• Embed   │  Rewrite │  Type    │• Context │• Temperature│
│• Loaders │  Model   │• HyDE    │• Hybrid  │  Compress│• Prompt Tmpl│
│• Cleaning│• VectorDB│• Query   │  Config  │• Context │• Self-RAG   │
│          │          │  Routing │• GraphRAG│  Assembly│• CRAG       │
│          │          │• Query   │• RAPTOR  │          │• FLARE      │
│          │          │  Decomp  │• Sm→Big  │          │• Agentic    │
│          │          │          │• SentWin │          │             │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘
                                    │
                              ┌─────┴─────┐
                              │  Module 7  │
                              │ EVALUATION │
                              │• RAGAS     │
                              │• DeepEval  │
                              │• Custom    │
                              └────────────┘
```

### Module 1: Data Ingestion
**Purpose**: Connect to data sources and parse documents into raw text.

| Component | Options | Parameters |
|-----------|---------|------------|
| Source Type | File upload, S3, GCS, URL, API, Database, SharePoint, GDrive | Connection credentials, refresh interval |
| Document Parser | Default, Azure Doc Intelligence, AWS Textract, Unstructured.io | Parser-specific settings |
| File Types | PDF, DOCX, TXT, HTML, Markdown, CSV, JSON, Code | Per-type parser configuration |
| Cleaning | Remove headers/footers, dedup, language filter | Enable/disable each |

### Module 2: Indexing
**Purpose**: Chunk documents, generate embeddings, and store in vector database.

| Component | Options | Parameters |
|-----------|---------|------------|
| Chunking Strategy | Fixed, Recursive, Semantic, Contextual, Page-level, Late | chunk_size, overlap, threshold, context_prompt |
| Embedding Model | OpenAI, Cohere, Voyage, BGE, Jina, Nomic, Gemini | model_name, dimensions, batch_size |
| Vector Database | ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, pgvector | connection_string, index_config, distance_metric |
| Metadata | Extract and store document metadata | fields_to_extract, custom_metadata |

### Module 3: Pre-Retrieval
**Purpose**: Transform and optimize the user query before retrieval.

| Component | Options | Parameters |
|-----------|---------|------------|
| Query Rewriting | Multi-query, Sub-question, Step-back | num_rewrites, rewrite_model |
| HyDE | Enable/Disable | num_hypotheses, hyde_model, temperature |
| Query Routing | Metadata, Semantic, Hybrid | routes[], confidence_threshold, fallback |
| Query Decomposition | Sequential, Parallel | max_sub_queries, aggregation_strategy |

### Module 4: Retrieval
**Purpose**: Find relevant documents using configured search strategies.

| Component | Options | Parameters |
|-----------|---------|------------|
| Search Type | Dense, Sparse, Hybrid | alpha, fusion_strategy, rrf_k |
| Top-K | Configurable | top_k per retriever |
| Advanced Retrieval | Small-to-Big, Sentence Window, GraphRAG, RAPTOR | Per-strategy parameters (see Section 2) |
| Contextual Retrieval | Anthropic method | context_model, use_bm25, top_k_retrieval |

### Module 5: Post-Retrieval
**Purpose**: Refine and compress retrieved context.

| Component | Options | Parameters |
|-----------|---------|------------|
| Reranking | Cohere, Cross-encoder, ColBERT, FlashRank, LLM-based | model, top_n, score_threshold |
| Context Compression | LLMLingua, Extractive, Sentence filter | compression_ratio, max_tokens |
| Context Assembly | Relevance-first, Chronological | separator, max_sources, source_attribution |

### Module 6: Generation
**Purpose**: Generate the final response using an LLM.

| Component | Options | Parameters |
|-----------|---------|------------|
| LLM Model | GPT-4o, Claude Opus/Sonnet, Gemini Pro, Llama, Mistral | model_name, api_key |
| Generation Config | Temperature, top_p, max_tokens | Standard LLM parameters |
| Prompt Template | System prompt, RAG prompt | Editable template with placeholders |
| Advanced | Self-RAG, CRAG, FLARE, Agentic | Per-technique parameters |

### Module 7: Evaluation
**Purpose**: Automated quality assessment of the generated pipeline.

| Component | Options | Parameters |
|-----------|---------|------------|
| Framework | RAGAS, DeepEval | metrics[], judge_model |
| Test Data | Upload, Auto-generate | num_test_cases, generation_model |
| Thresholds | Per-metric pass/fail | threshold per metric |
| Monitoring | Continuous eval in production | eval_frequency, alert_threshold |

---

## 4. System Architecture Proposal

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React/Next.js)                       │
│                                                                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Pipeline    │  │ Module       │  │ Config       │  │ Pipeline         │  │
│  │ Canvas     │  │ Configurator │  │ Preview      │  │ Monitor/Eval     │  │
│  │ (Visual)   │  │ (Forms)      │  │ (YAML/JSON)  │  │ (Dashboard)      │  │
│  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│        │                │                  │                   │             │
│        └────────────────┴──────────────────┴───────────────────┘             │
│                                    │                                        │
│                          Pipeline Config JSON                               │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ REST API / WebSocket
┌────────────────────────────────────┴────────────────────────────────────────┐
│                           BACKEND (Python / FastAPI)                         │
│                                                                             │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────────────────┐  │
│  │  Config          │  │  Code           │  │  Deployment                  │  │
│  │  Validator       │  │  Generator      │  │  Engine                      │  │
│  │                  │  │                 │  │                              │  │
│  │ • Schema valid.  │  │ • Template eng. │  │ • Docker build               │  │
│  │ • Compat check   │  │ • LangChain gen │  │ • K8s deploy                 │  │
│  │ • Cost estimate  │  │ • LlamaIndex gen│  │ • Serverless (Lambda/Cloud)  │  │
│  │ • Perf estimate  │  │ • Test gen      │  │ • Health monitoring          │  │
│  └────────┬─────────┘  └───────┬────────┘  └──────────────┬───────────────┘  │
│           │                    │                           │                 │
│  ┌────────┴────────────────────┴───────────────────────────┴──────────────┐  │
│  │                      Pipeline Registry (PostgreSQL)                     │  │
│  │  • Pipeline configs  • Generated code  • Deploy status  • Eval results │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Configuration-First Approach

The **intermediate representation** between UI and code is a structured YAML/JSON config. This is the single source of truth.

```yaml
# Example: Generated pipeline config
version: "1.0"
pipeline:
  name: "legal-document-rag"
  tier: "advanced"

  ingestion:
    sources:
      - type: "s3"
        bucket: "legal-docs"
        file_types: ["pdf", "docx"]
    parser: "azure_doc_intelligence"

  indexing:
    chunking:
      strategy: "contextual"
      chunk_size: 800
      chunk_overlap: 80
      context_model: "claude-haiku"
    embedding:
      provider: "voyage"
      model: "voyage-3-large"
      dimensions: 1024
    vector_db:
      provider: "qdrant"
      collection: "legal-chunks"
      distance_metric: "cosine"

  pre_retrieval:
    query_rewriting:
      enabled: true
      strategy: "multi_query"
      num_rewrites: 3
    hyde:
      enabled: false
    routing:
      enabled: true
      routes:
        - name: "case_law"
          description: "Questions about legal precedents and case law"
        - name: "regulatory"
          description: "Questions about regulations and compliance"

  retrieval:
    search_type: "hybrid"
    fusion_strategy: "rrf"
    rrf_k: 60
    top_k: 100
    contextual_retrieval:
      enabled: true
      use_bm25: true

  post_retrieval:
    reranking:
      enabled: true
      model: "cohere-rerank-3.5"
      top_n: 10
    context_compression:
      enabled: false
    context_assembly:
      ordering: "relevance_first"
      max_sources: 5
      source_attribution: true

  generation:
    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"
      temperature: 0.05
      max_tokens: 2048
    prompt_template: |
      You are a legal research assistant. Answer based ONLY on the provided context.
      If the answer is not in the context, say "I cannot find this information in the provided documents."

      Context:
      {context}

      Question: {query}
    advanced:
      crag:
        enabled: true
        upper_threshold: 0.7
        web_search_fallback: false

  evaluation:
    framework: "ragas"
    metrics: ["faithfulness", "answer_relevancy", "context_precision"]
    judge_model: "gpt-4o"
    pass_threshold: 0.75
```

### 4.3 Code Generation Engine

The Code Generator translates YAML config into executable pipeline code. **Recommended approach: Jinja2 template engine** with framework-specific templates.

```
pipeline_config.yaml
        │
        ▼
┌─────────────────────┐
│  Template Selector   │  ← Chooses LangChain or LlamaIndex templates
├─────────────────────┤
│  Jinja2 Renderer     │  ← Fills templates with config values
├─────────────────────┤
│  Dependency Resolver │  ← Generates requirements.txt / pyproject.toml
├─────────────────────┤
│  Code Validator      │  ← AST parse + type check generated code
├─────────────────────┤
│  Test Generator      │  ← Creates integration tests from eval config
└─────────────────────┘
        │
        ▼
  project/
  ├── src/
  │   ├── pipeline.py          # Main pipeline code
  │   ├── ingestion.py         # Data loading
  │   ├── retriever.py         # Search configuration
  │   ├── generator.py         # LLM + prompt
  │   └── config.py            # Runtime config from env vars
  ├── tests/
  │   ├── test_retrieval.py    # Retrieval quality tests
  │   └── test_generation.py   # Generation quality tests
  ├── Dockerfile
  ├── docker-compose.yml
  ├── pyproject.toml
  └── README.md
```

### 4.4 Technology Stack Recommendation

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Frontend** | Next.js + React + TypeScript | Modern, fast, great for form-heavy UIs |
| **UI Components** | Shadcn/ui + React Flow | React Flow for visual pipeline canvas |
| **Backend API** | FastAPI (Python) | Async, typed, Python ecosystem (same as LangChain/LlamaIndex) |
| **Config Schema** | Pydantic models | Automatic validation, JSON Schema generation for frontend |
| **Code Generation** | Jinja2 templates | Battle-tested, flexible, auditable output |
| **Target Frameworks** | LangChain + LlamaIndex | Recommended hybrid: LlamaIndex for ingestion/indexing, LangChain/LangGraph for orchestration |
| **Database** | PostgreSQL | Pipeline configs, user data, eval results |
| **Deployment** | Docker + optional K8s | Containerized pipelines, scalable |
| **Eval** | RAGAS + DeepEval | Automated quality gates |

### 4.5 Key Design Decisions

1. **Generate code, not just run configs**: Unlike Dify/Flowise that execute workflows at runtime, we generate standalone Python projects. Developers own and can modify the output. This is our core differentiator.

2. **Framework-agnostic config, framework-specific output**: The YAML config is framework-agnostic. Templates exist for LangChain, LlamaIndex, or even raw Python. Users choose their target framework.

3. **Progressive disclosure in UI**: Default to "Recommended" presets (Advanced RAG with hybrid search + reranking). Power users expand each module for fine-grained control.

4. **Cost estimation**: Before generating, the backend estimates indexing cost (embedding API calls, contextual chunking LLM calls, GraphRAG extraction) and per-query cost (LLM generation, reranking API). Display prominently in UI.

5. **Compatibility matrix**: Not all combinations work. Enforce constraints:
   - Self-RAG requires specific model → gray out incompatible LLMs
   - FLARE requires logprobs → gray out providers that don't support it
   - GraphRAG + real-time latency → show warning
   - Contextual chunking + large corpus → show cost warning

---

## 5. Blind Spots & Edge Cases

### 5.1 Critical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Combinatorial explosion** | HIGH | Not all 2^N feature combinations are tested. Implement a compatibility matrix and restrict untested combinations with warnings. |
| **Benchmark ≠ production** | HIGH | MTEB scores don't predict domain performance. Mandate evaluation on user's actual data before deployment. Include an "Evaluate Before Deploy" gate. |
| **Hidden cost explosions** | HIGH | GraphRAG indexing, contextual chunking, and agentic RAG can generate thousands of LLM calls. Always surface cost estimates before execution. |
| **Context window misuse** | MEDIUM | Models reliably use only 50-65% of advertised context (NVIDIA RULER). Don't let users assume they can stuff 200K tokens into Claude and get consistent quality. Default max_context_tokens conservatively. |
| **Stale embeddings** | MEDIUM | When corpus updates, old embeddings become stale. Implement incremental re-indexing and warn users about embedding model changes (changing models requires full re-indexing). |
| **Table/structure destruction** | MEDIUM | PDFs lose spatial meaning during extraction. Layout-aware parsers (Azure Doc Intelligence, Textract) are expensive but necessary for structured docs. Surface this as a parser choice with quality trade-off. |

### 5.2 Edge Cases in Automation

1. **Domain-specific embedding fine-tuning**: General embeddings may fail in specialized domains (medical, legal). The platform should support plugging in fine-tuned embedding models, not just off-the-shelf ones.

2. **Multilingual corpora**: Chunk boundaries, BM25 tokenization, and embedding quality vary by language. If the user specifies multiple languages, recommend multilingual models (BGE-M3, Cohere embed-v4) and warn about BM25 limitations.

3. **Multi-modal documents**: Images, charts, and diagrams in PDFs are lost in text extraction. Future: integrate vision models for image-to-text during ingestion.

4. **Compliance and data residency**: Some users can't send data to external APIs. The platform should distinguish between cloud-API components (OpenAI, Cohere) and self-hostable components (BGE, Qdrant, Llama). Display data-flow diagrams showing where data leaves the user's infrastructure.

5. **Evaluation without ground truth**: RAGAS claims "reference-free" evaluation, but Context Precision and Context Recall still require ground truth. Either auto-generate test QA pairs from the corpus (using an LLM) or clearly mark which metrics need ground truth.

### 5.3 What Cannot Be Fully Automated

| Component | Why It Resists Automation | Recommended Approach |
|-----------|--------------------------|---------------------|
| **Chunking strategy selection** | Optimal strategy depends on document structure, which varies per corpus | Provide "analyze sample" feature that tests multiple strategies and recommends |
| **Prompt engineering** | Domain-specific instructions require human judgment | Provide templates per domain + prompt playground |
| **Confidence thresholds** | CRAG thresholds are dataset-specific (e.g., PopQA: 0.59, Biography: 0.95) | Default conservatively, expose in advanced settings, recommend tuning via eval |
| **GraphRAG entity schema** | Which entities/relationships to extract depends on domain | Offer "auto-extract" with manual refinement |
| **Routing rules** | Requires knowledge of data source contents | Provide guided setup wizard with example queries per route |
| **Self-RAG model training** | Requires fine-tuning infrastructure | Offer pre-trained models only; link to fine-tuning guide for advanced users |

### 5.4 Recommended Risk Mitigation Architecture

```
User Config → Compatibility Checker → Cost Estimator → Code Generator →
  → Static Analysis → Generated Tests → Sandbox Deploy → Eval Gate → Production
```

Every pipeline goes through:
1. **Compatibility check**: Reject invalid combinations before generation
2. **Cost estimate**: Surface expected indexing + per-query costs
3. **Static analysis**: Verify generated code compiles and passes linting
4. **Sandbox deployment**: Run generated pipeline against sample data
5. **Evaluation gate**: Run RAGAS/DeepEval, require minimum scores before production deployment

---

## Appendix A: Decision Matrix — All Techniques

| Technique | Complexity | Latency Impact | Quality Gain | Automation Feasibility | Priority for MVP |
|-----------|-----------|----------------|-------------|----------------------|-----------------|
| Hybrid Search | Low-Med | Low | 15-30% recall | **HIGH** | **P0 — Must have** |
| Reranking | Low-Med | Medium | Significant | **HIGH** | **P0 — Must have** |
| Chunking Strategies | Low-Med | Low (indexing) | ~70% lift | **HIGH** | **P0 — Must have** |
| Embedding Selection | Low | Low | Varies by domain | **HIGH** | **P0 — Must have** |
| Vector DB Selection | Low | Low | Infra decision | **HIGH** | **P0 — Must have** |
| LLM Selection | Low | Low | Core function | **HIGH** | **P0 — Must have** |
| Prompt Template | Low | None | High | **HIGH** | **P0 — Must have** |
| Query Rewriting | Medium | Medium | Large for complex | **HIGH** | **P1 — Should have** |
| HyDE | Medium | Medium | Variable | **HIGH** | **P1 — Should have** |
| Contextual Retrieval | Medium | Med (indexing) | 49-67% fewer failures | **HIGH** | **P1 — Should have** |
| CRAG | High | Medium-High | ~7% over baseline | **MED-HIGH** | **P1 — Should have** |
| Small-to-Big | Medium | Low | Good | **HIGH** | **P1 — Should have** |
| Sentence Window | Medium | Low | Good | **HIGH** | **P1 — Should have** |
| Context Compression | Medium | Medium | Moderate | **MED-HIGH** | **P2 — Nice to have** |
| Query Routing | Medium | Medium | Large for multi-source | **MEDIUM** | **P2 — Nice to have** |
| Query Decomposition | Medium | Medium | Large for complex | **MEDIUM** | **P2 — Nice to have** |
| Agentic RAG | Very High | High | 44% on complex | **MEDIUM** | **P2 — Nice to have** |
| RAPTOR | High | High (indexing) | 20% on multi-hop | **MEDIUM** | **P3 — Future** |
| GraphRAG | Very High | Very High | Large for global | **MEDIUM** | **P3 — Future** |
| FLARE | High | High | Good for long-form | **MEDIUM** | **P3 — Future** |
| Self-RAG | Very High | Medium | Significant | **LOW** | **P3 — Future** |
| Evaluation (RAGAS) | Low-Med | N/A | Quality assurance | **HIGH** | **P0 — Must have** |

---

## Appendix B: Embedding Model Quick Reference (2025)

| Model | MTEB | Context | Dims | Cost/1M Tokens | Self-Host | Best For |
|-------|------|---------|------|----------------|-----------|----------|
| Qwen3-Embedding-8B | 70.58 | 32K | 7,168 | Free | Yes | Highest quality, multilingual |
| Gemini embedding-001 | 68.32 | 2K | 3,072 | $0.15 | No | Google ecosystem |
| Voyage-3-large | ~67+ | 32K | 2,048 | $0.06 | No | Best API value |
| Cohere embed-v4 | 65.2 | 128K | 1,024 | $0.10 | No | Longest context window |
| OpenAI text-embedding-3-large | 64.6 | 8K | 3,072 | $0.13 | No | Broadest ecosystem |
| BGE-M3 | 63.0 | 8K | 1,024 | Free | Yes | Hybrid (dense+sparse+multi-vec) |
| Jina v3 | ~62+ | 8K | 1,024 | $0.018 | Yes* | Cheapest API |
| Nomic embed-text-v1.5 | ~62+ | 8K | 768 | Free | Yes | Fully open (weights+data+code) |

## Appendix C: Vector Database Quick Reference (2025)

| Database | Type | Best For | Filtering | Scale | Starting Cost |
|----------|------|----------|-----------|-------|--------------|
| **ChromaDB** | OSS | Prototyping | Basic | Small | Free |
| **pgvector** | Postgres ext | Existing Postgres users | Full SQL | Medium | Free |
| **Qdrant** | OSS (Rust) | Complex metadata filtering | Best-in-class | Large | Free / $30/mo cloud |
| **Weaviate** | OSS + Cloud | Knowledge graph + semantic | GraphQL | Medium | Free / $75/mo cloud |
| **Pinecone** | Managed SaaS | Zero-ops, fast setup | Strong | Auto | $96/mo+ |
| **Milvus** | OSS + Managed | Billion-scale vectors | Comprehensive | Very Large | Free / $0.10/hr |

---

*This report was compiled from web research across 40+ sources including arXiv papers, framework documentation, and industry benchmarks. All benchmarks cited are from 2024-2025 publications.*
