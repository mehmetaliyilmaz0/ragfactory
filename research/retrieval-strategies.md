# RAG Retrieval Strategies: Exhaustive Research

> Research compiled: 2026-04-04
> Sources: Academic papers, vendor documentation, production case studies

---

## Table of Contents

1. [Dense Retrieval](#1-dense-retrieval)
2. [Sparse Retrieval (BM25)](#2-sparse-retrieval-bm25)
3. [Hybrid Search (Dense + Sparse)](#3-hybrid-search-dense--sparse)
4. [Small-to-Big Retrieval (Parent-Child)](#4-small-to-big-retrieval-parent-child)
5. [Sentence Window Retrieval](#5-sentence-window-retrieval)
6. [Contextual Retrieval (Anthropic)](#6-contextual-retrieval-anthropic)
7. [Multi-Vector Retrieval (ColBERT)](#7-multi-vector-retrieval-colbert)
8. [Ensemble Retrieval](#8-ensemble-retrieval)
9. [Retrieval Evaluation Metrics](#9-retrieval-evaluation-metrics)
10. [Production Considerations](#10-production-considerations)

---

## 1. Dense Retrieval

### 1.1 Bi-Encoder Architecture

Dense retrieval encodes queries and documents into a shared semantic vector space using neural networks. The **bi-encoder** (dual encoder) architecture is the standard approach:

**How it works:**
- Two separate encoder towers (often sharing weights) process query and document independently
- Each encoder produces a fixed-dimensional embedding vector (typically 384, 768, or 1024 dimensions)
- A pooling layer (CLS token or mean pooling) collapses token representations into a single vector
- At inference time, only the similarity between two pre-computed vectors needs to be calculated

**Key advantage:** Documents can be pre-encoded offline; only the query needs encoding at search time.

**Key limitation:** No cross-attention between query and document tokens means the model cannot capture fine-grained token-level interactions. This is the fundamental performance bottleneck compared to cross-encoders.

**Modern evolution (2024-2025):** LLM-based embedding models (e.g., text-embedding-3-large, Voyage, Gemini Text) now dominate BEIR and MTEB benchmarks, leveraging superior world knowledge and semantic understanding from billion-parameter backbones.

### 1.2 Embedding Similarity Metrics

| Metric | Formula | Best For | Notes |
|--------|---------|----------|-------|
| **Cosine Similarity** | `cos(A,B) = (A . B) / (\|A\| * \|B\|)` | Text embeddings (most common) | Direction-only; ignores magnitude. Default for most embedding APIs |
| **Dot Product** | `A . B = sum(a_i * b_i)` | Recommendation systems | Captures both direction and magnitude. Magnitude encodes confidence/importance |
| **Euclidean Distance** | `\|A - B\| = sqrt(sum((a_i - b_i)^2))` | Lower-dimensional spaces | Magnitude-sensitive. Suffers from curse of dimensionality at 768+ dims |

**Practical note:** For L2-normalized vectors (unit vectors), cosine similarity and dot product produce identical rankings. Most modern embedding models output normalized vectors, so the choice is often academic.

### 1.3 HNSW (Hierarchical Navigable Small Worlds)

The dominant ANN (Approximate Nearest Neighbor) index for production vector search.

**How it works:**
- Builds a multi-layer graph where each node is a vector
- Top layers have few, long-range connections (coarse navigation)
- Bottom layers have many, short-range connections (precise search)
- Search starts at the top layer entry point, greedily traverses to local minimum, then drops down
- Layer assignment is probabilistic: `P(layer) = exp(-level / m_L) * (1 - exp(-1 / m_L))` where `m_L = 1/ln(M)`

**Parameters:**

| Parameter | What It Controls | Range | Default | Trade-off |
|-----------|-----------------|-------|---------|-----------|
| **M** | Max bidirectional links per node | 2-100 | 16 | Higher M = better recall, more memory. M=16 works for most cases; M=64+ for high-dimensional data |
| **ef_construction** | Candidate list size during build | 100-500 | 200 | Higher = better graph quality, slower build. Does NOT affect index size |
| **ef_search** | Candidate list size during query | 10-500 | 100 | Higher = better recall, higher latency. Tunable at query time without rebuild |

**Memory scaling:** Depends exclusively on M. With Sift1M dataset: M=2 requires ~0.5GB, M=512 reaches ~5GB.

**Recall vs. latency:** Recall ranges from ~80% (low ef_search) to 99.9%+ (high ef_search). Single-query latency ranges from <1ms to ~50ms depending on parameters and dataset size.

**Recommended starting points:**
- General purpose: M=16, ef_construction=200, ef_search=100
- High-recall requirement: M=32-64, ef_construction=400, ef_search=200-400
- Memory-constrained: M=8-12, ef_construction=100, ef_search=64

### 1.4 IVF (Inverted File Index)

An alternative to HNSW that partitions the vector space into Voronoi cells.

**How it works:**
1. K-means clustering divides vectors into `nlist` clusters (cells)
2. Each vector is assigned to its nearest centroid
3. At query time, the `nprobe` nearest centroids are identified, and only vectors in those cells are searched

**Parameters:**

| Parameter | What It Controls | Typical Values | Notes |
|-----------|-----------------|----------------|-------|
| **nlist** | Number of clusters | 256-65536 (powers of 2) | Fixed at build time; rebuild required to change. Rule of thumb: sqrt(N) to 4*sqrt(N) |
| **nprobe** | Clusters searched at query time | 1-nlist (typically 8-64) | Tunable without rebuild. Higher = better recall, more latency |

**IVF vs HNSW trade-offs:**
- IVF: Lower memory, faster build, better for billion-scale with disk
- HNSW: Better recall-latency ratio, faster queries, higher memory

### 1.5 Product Quantization (PQ)

Compresses vectors for memory reduction, often combined with IVF (IVFPQ).

**How it works:**
1. Split a D-dimensional vector into `m` sub-vectors of `D/m` dimensions each
2. Train a codebook of 256 entries per sub-space (via k-means)
3. Replace each sub-vector with its nearest codebook index (8-bit)

**Example compression:** A 128-dim float32 vector (512 bytes) with m=8 sub-spaces becomes 8 bytes total -- a 64:1 compression ratio.

**Trade-off:** Significant memory savings at the cost of recall degradation. IVFPQ is essential for billion-scale datasets where full vectors cannot fit in memory.

### 1.6 Dense Retrieval Failure Modes

| Failure Mode | Description | Mitigation |
|--------------|-------------|------------|
| **Semantic Drift** | Encoder parameters drift, causing mismatches between query and document representations | Regular model evaluation; hybrid search |
| **Vocabulary Gap** | Domain-specific jargon, acronyms, or rare terms not well-represented in embedding space | Fine-tuning on domain data; hybrid with BM25 |
| **Ambiguous Queries** | Short queries (avg <2.5 words in web search) produce averaged representations that dilute distinct interpretations | Query expansion; multi-vector representations |
| **Granularity Dilemma** | Training for fine-grained entity matching hurts broad semantic understanding, and vice versa | Multi-stage retrieval; ensemble approaches |
| **Out-of-Distribution** | Queries or documents far from training distribution produce unreliable embeddings | Domain adaptation; BEIR-style evaluation |

---

## 2. Sparse Retrieval (BM25)

### 2.1 BM25 Mathematical Formulation

The complete BM25 scoring function:

```
Score(D, Q) = SUM over qi in Q of:
    IDF(qi) * [ f(qi, D) * (k1 + 1) ] / [ f(qi, D) + k1 * (1 - b + b * |D| / avgdl) ]
```

Where:
- `qi` = individual query term
- `f(qi, D)` = frequency of term qi in document D
- `|D|` = document length (in tokens)
- `avgdl` = average document length across corpus
- `k1` = term frequency saturation parameter
- `b` = document length normalization parameter

**IDF component (Lucene/ES variant):**
```
IDF(qi) = log[ (docCount - f(qi) + 0.5) / (f(qi) + 0.5) ]
```

Where `docCount` = total documents, `f(qi)` = documents containing term qi.

### 2.2 TF-IDF vs BM25 Differences

| Aspect | TF-IDF | BM25 |
|--------|--------|------|
| **Term frequency** | Linear growth (unlimited) | Saturates via k1 parameter (diminishing returns) |
| **Length normalization** | Basic or none | Adaptive via b parameter |
| **Parameterization** | Fixed calculation | Tunable via k1, b |
| **Score behavior** | TF grows linearly forever | TF curve approaches asymptote |

### 2.3 BM25 Parameters Deep Dive

**k1 (Term Frequency Saturation) -- Default: 1.2**

| k1 Value | Behavior | Use Case |
|----------|----------|----------|
| 0 | Only IDF matters; TF ignored entirely | When term presence/absence is all that matters |
| 0.5-1.0 | Quick saturation; diminishing returns early | Short documents, titles |
| **1.2** | **Standard default** | **General purpose** |
| 1.5-2.0 | Slower saturation; repeated terms matter more | Long documents, technical content |
| 5-10 | Near-linear TF growth | Extreme frequency differentiation |

**b (Length Normalization) -- Default: 0.75**

| b Value | Behavior | Use Case |
|---------|----------|----------|
| 0 | Document length completely ignored | All documents similar length |
| 0.5 | Moderate length penalty | Mixed-length collections |
| **0.75** | **Standard default** | **General purpose** |
| 1.0 | Maximum length penalization | Strongly varied document lengths |

**Practical guidance from Elastic:** The defaults of k1=1.2 and b=0.75 work well for most corpora. Only tune when you have evaluation data showing suboptimal retrieval.

### 2.4 SPLADE: Learned Sparse Representations

SPLADE (Sparse Lexical and Expansion Model) bridges the gap between sparse and dense retrieval.

**Architecture:**
1. Input text passes through BERT with the MLM (Masked Language Model) head
2. For each token position, the MLM head produces logits over the full 30,522-token vocabulary
3. A log-saturation transform is applied: `log(1 + ReLU(logits)) * attention_mask`
4. Max pooling across all token positions produces one importance weight per vocabulary term
5. Non-zero weights form the sparse vector representation

**Term Expansion (Critical Innovation):**
When processing a word like "rainforest," the MLM head predicts related terms (jungle, forest, tropical) with associated weights. These expanded terms populate the sparse vector, directly addressing the vocabulary mismatch problem that plagues BM25.

**FLOPS Regularization:**
SPLADE uses a sparsity regularizer based on FLOPS (Floating Point Operations Per Second) to control the number of non-zero values. This prevents the model from producing dense-like representations while maintaining enough expansion for semantic coverage.

### 2.5 SPLADEv2 and SPLADE++ Improvements

**SPLADEv2:**
- Hard-negative mining during training
- Knowledge distillation from cross-encoders
- Better PLM (Pre-trained Language Model) initialization
- Max pooling mechanism improvement
- SPLADE-doc variant: document-only encoder that pre-computes term weights, eliminating query-side inference cost

**SPLADE++:**
- Expands both documents AND queries at inference time (unlike SPLADE-doc)
- Benefits from the same training improvements as dense bi-encoders (distillation, hard negatives)
- State-of-the-art results on both in-domain (MS MARCO) and out-of-domain (BEIR) benchmarks

**Limitations:**
- Slower than traditional BM25 due to more non-zero values in representations
- Models trained on MS MARCO can generalize poorly to other domains (may underperform BM25)
- Most systems lack native SPLADE support (though Qdrant, OpenSearch, and others are adding it)
- Model size: ~532 MB for SPLADE++ (BERT-base backbone)

### 2.6 When Sparse Beats Dense

| Scenario | Why Sparse Wins | Example |
|----------|----------------|---------|
| **Exact entity matching** | Dense embeddings average away proper nouns | "John Smith invoice 2024" |
| **Rare/specialized terms** | Out-of-vocabulary for embedding models | Medical codes, legal citations |
| **Acronyms and identifiers** | Dense models undertrained on these patterns | "HIPAA", "CVE-2024-1234" |
| **Short keyword queries** | Insufficient context for semantic understanding | "pandas groupby agg" |
| **Exact phrase requirements** | Dense cannot guarantee exact phrase presence | Legal clause lookup |

### 2.7 BM25 Implementation Platforms

| Platform | Implementation | Notes |
|----------|---------------|-------|
| **Elasticsearch** | Lucene BM25 (default since ES 5.0) | BM25F for field-weighted scoring. Most mature ecosystem |
| **OpenSearch** | Lucene BM25 + neural sparse search | v2.12+ enhanced sparse search throughput/latency |
| **Apache Lucene** | Native BM25Similarity | Foundation for ES, OpenSearch, Solr |
| **Tantivy** | BM25 (Rust-based) | Used by Qdrant and other Rust vector DBs. Fast, lightweight |
| **Milvus 2.5+** | BM25 scores as sparse vectors | Native alongside dense embeddings |

---

## 3. Hybrid Search (Dense + Sparse)

### 3.1 Architecture Patterns

Hybrid search runs both dense (semantic) and sparse (lexical) retrieval in parallel, then fuses results. Two primary fusion strategies exist:

**Pattern 1: Reciprocal Rank Fusion (RRF)**
- Score-agnostic; works purely on rank positions
- No score normalization needed

**Pattern 2: Weighted Linear Combination**
- Normalizes scores from each retriever to a common scale
- Combines with weighted sum

### 3.2 Reciprocal Rank Fusion (RRF)

**Mathematical Formulation:**
```
RRF_score(d) = SUM over r in R of: 1 / (k + rank_r(d))
```

Where:
- `d` = document being scored
- `R` = set of all ranking systems (retrievers)
- `k` = smoothing constant (default: 60)
- `rank_r(d)` = position of document d in ranker r's results (1-indexed)

**k Parameter (Default: 60):**
- Prevents any single top-ranked result from dominating
- Dampens the difference between rank 1 and rank 2
- At k=60: rank 1 yields ~0.0164, rank 10 yields ~0.0143, rank 100 yields ~0.00625
- Lower k (e.g., 10) amplifies rank differences; higher k (e.g., 100) flattens them

**Pros:**
- Zero-shot: no training or tuning needed
- Robust across diverse datasets (outperforms Condorcet and learned methods)
- No score normalization required
- Simple to implement and debug

**Cons:**
- Ignores score magnitude (a result barely making top-10 is treated same as a dominant #10)
- Fixed k may not be optimal for all query types
- Cannot leverage high-confidence scores from individual retrievers

### 3.3 Weighted Linear Combination

**Formula:**
```
H(d) = (1 - alpha) * K(d) + alpha * V(d)
```

Where:
- `H(d)` = hybrid score
- `alpha` = weight parameter (0 to 1)
- `K(d)` = normalized keyword/BM25 score
- `V(d)` = normalized vector/dense score

When alpha=1.0: pure vector search. When alpha=0.0: pure keyword search.

**Score Normalization Methods:**
1. **Min-Max normalization**: `score_norm = (score - min) / (max - min)` -- Simple but sensitive to outliers
2. **Relative Score Fusion (Weaviate)**: Preserves original score distributions while normalizing to [0,1] range
3. **Z-score normalization**: `score_norm = (score - mean) / stddev` -- More robust to outliers

### 3.4 Alpha/Weight Tuning Methodology

1. Start with alpha=0.5 (equal weight) as baseline
2. Evaluate on a labeled dataset using Recall@K or NDCG@K
3. Grid search alpha in [0.0, 0.1, 0.2, ..., 1.0]
4. Typical findings:
   - Technical/code domains: alpha 0.3-0.5 (favor keyword matching)
   - Conversational/semantic domains: alpha 0.6-0.8 (favor dense)
   - Mixed-domain production: alpha 0.5-0.7

**When to retune:** After significant corpus changes, new document types, or embedding model upgrades.

### 3.5 Three-Way Hybrid (BM25 + Dense + SPLADE)

Research (including IBM work) shows combining three retrieval signals can outperform two-way hybrid:
- BM25 catches exact lexical matches
- Dense captures semantic similarity
- SPLADE bridges the gap with learned sparse expansion

Implementation: Run all three retrievers, apply RRF or weighted fusion across three ranked lists.

### 3.6 Implementation by Vector Database

| Database | Hybrid Approach | Fusion Method | Key Feature |
|----------|----------------|---------------|-------------|
| **Weaviate** | Native BM25F + vector | Alpha parameter + Relative Score Fusion (default) or RRF | Most mature native implementation. BlockMax WAND makes keyword side 10x faster |
| **Qdrant** | Named vectors (dense HNSW + sparse inverted index) | Prefetch + re-score in Universal Query API | Multi-stage architectures in single request. IDF computation server-side since v1.15.2 |
| **Pinecone** | Proprietary sparse-dense | Weighted combination | Locked into Pinecone's sparse format (not standard BM25/SPLADE) |
| **Milvus 2.5+** | BM25 as sparse vectors + dense embeddings | RRF or weighted | Claims 30x latency advantage over Elasticsearch |
| **Elasticsearch** | Native BM25 + kNN vector search | RRF (since ES 8.8) or scripted scoring | Leverages legacy lexical search strength |
| **OpenSearch** | BM25 + neural search | RRF (since OS 2.12) | Neural sparse search optimized in Lucene engine |
| **pgvector + pg_bm25** | Separate indexes, application-level fusion | Application code | Most flexible but requires custom fusion logic |

### 3.7 RRF vs Weighted: When to Use Which

| Criterion | Use RRF | Use Weighted |
|-----------|---------|-------------|
| No labeled evaluation data | Yes (zero-shot) | No (needs tuning data) |
| Heterogeneous score distributions | Yes (rank-based) | Requires normalization |
| Want to leverage high-confidence scores | No | Yes |
| Multiple (3+) retrievers | Yes (scales easily) | Complex weight tuning |
| Single domain, stable queries | Either | Yes (can optimize) |
| Need explainability of fusion | Yes (simple formula) | Yes (transparent weights) |

---

## 4. Small-to-Big Retrieval (Parent-Child)

### 4.1 Core Concept

Small-to-Big retrieval decouples the retrieval unit from the synthesis unit:
- **Small chunks** (128-512 tokens) are embedded and used for vector similarity search
- **Big chunks** (1024-2048 tokens) or full documents are returned to the LLM for generation

This addresses the fundamental tension: small chunks produce better embeddings (less filler text diluting semantics), but LLMs need broader context for quality generation.

### 4.2 LangChain ParentDocumentRetriever

**Implementation:**
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define child and parent splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()  # Stores parent documents
vectorstore = ...  # Your vector store for child embeddings

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,  # Optional: if None, uses full document as parent
)
```

**How parent references are stored:**
- Child chunks are embedded and stored in the vectorstore
- Each child chunk's metadata contains a `parent_doc_id` linking to the parent
- Parent documents are stored in a separate `docstore` (InMemoryStore, Redis, etc.)
- During retrieval: child chunks are fetched by similarity, then parent_doc_ids are looked up, and full parent documents are returned

**Two modes:**
1. **Full document as parent:** Set `parent_splitter=None`. Returns the entire original document.
2. **Larger chunk as parent:** Provide both splitters. Returns the parent chunk containing the matched child.

### 4.3 LlamaIndex Recursive Retriever (Child-Parent)

**Implementation:**
```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever

# Create parent nodes (1024 tokens)
parent_parser = SentenceSplitter(chunk_size=1024)
parent_nodes = parent_parser.get_nodes_from_documents(documents)

# Create child nodes (256 tokens) linked to parents
child_parser = SentenceSplitter(chunk_size=256)
child_nodes = []
for parent in parent_nodes:
    children = child_parser.get_nodes_from_documents([parent.to_document()])
    for child in children:
        index_node = IndexNode(
            text=child.text,
            index_id=parent.node_id,  # Link to parent
        )
        child_nodes.append(index_node)
```

The `RecursiveRetriever` traverses node relationships, following links from child nodes to parent nodes, ensuring the LLM receives full context.

### 4.4 Chunk Size Optimization

| Child Size | Parent Size | Use Case | Trade-off |
|-----------|-------------|----------|-----------|
| 128 tokens | 1024 tokens | Fine-grained factoid retrieval | High precision, many chunks to manage |
| 256 tokens | 1024 tokens | General purpose (recommended starting point) | Good balance |
| 512 tokens | 2048 tokens | Analytical/long-form content | Fewer chunks, broader context |
| Single sentence | Full section/document | Maximum retrieval precision | Highest storage overhead |

### 4.5 Storage and Memory Implications

- **Storage overhead:** 2x-3x vs standard chunking (child embeddings + parent document store)
- **Parent overlap:** When parent chunks overlap, the same content may be returned multiple times. Deduplication is needed at the application level.
- **Docstore choice matters:** InMemoryStore is fast but limited by RAM. Redis or a database is needed for large corpora.

---

## 5. Sentence Window Retrieval

### 5.1 How It Works

Sentence Window Retrieval is a specialized form of small-to-big retrieval where:
1. Documents are parsed into individual sentences
2. Each sentence is embedded separately (maximum retrieval precision)
3. Surrounding sentences (the "window") are stored as metadata
4. At retrieval time, the matched sentence is replaced with its full window before being sent to the LLM

### 5.2 LlamaIndex Implementation

**SentenceWindowNodeParser:**
```python
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # Number of sentences on each side
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodes = node_parser.get_nodes_from_documents(documents)
```

**MetadataReplacementNodePostProcessor:**
```python
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)
# Applied after retrieval, before passing to LLM
```

**Sentence boundary detection:** Uses NLTK's `sent_tokenize` or similar sentence splitters. Quality depends heavily on the tokenizer (works well for standard English prose, may struggle with code, tables, or non-standard formatting).

### 5.3 Window Size Optimization

| Window Size | Total Context | Best For | Trade-off |
|-------------|---------------|----------|-----------|
| 1 (3 sentences total) | Minimal context | Factoid Q&A | May miss important surrounding context |
| **3 (7 sentences total)** | **Moderate (recommended)** | **General purpose** | **Good balance of precision and context** |
| 5 (11 sentences total) | Substantial context | Analytical questions | Larger context may include irrelevant info |
| 7-10 | Very large context | Complex reasoning | Approaches parent-child chunk sizes; diminishing returns |

### 5.4 Performance Comparison: Sentence Window vs Small-to-Big

| Aspect | Sentence Window | Small-to-Big (Parent-Child) |
|--------|----------------|---------------------------|
| **Retrieval granularity** | Single sentence (finest) | 128-512 token chunks |
| **Context delivery** | Sliding window around match | Full parent chunk |
| **Storage overhead** | Higher (window metadata per sentence) | Moderate (parent docstore) |
| **Implementation complexity** | Simpler (single parser + postprocessor) | More complex (two splitters + docstore) |
| **Best for** | Dense documents with varied topics per section | Documents with logical section boundaries |
| **Overlap handling** | Natural (windows overlap by design) | Must be managed explicitly |

---

## 6. Contextual Retrieval (Anthropic)

### 6.1 Core Concept

Anthropic's Contextual Retrieval prepends chunk-specific explanatory context to each chunk before embedding and BM25 indexing. An LLM generates this context by seeing both the full document and the specific chunk.

### 6.2 Exact Prompt Template

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk. Answer
only with the succinct context and nothing else.
```

**Output:** 50-100 tokens of contextual text, prepended to the chunk.

**Example transformation:**
- Original: "The company's revenue grew by 3% over the previous quarter."
- Contextualized: "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."

### 6.3 Cost Analysis

Assuming 800-token chunks, 8K-token documents, 50-token instructions, 100-token context output:

| Corpus Size | Without Prompt Caching | With Prompt Caching | Notes |
|-------------|----------------------|--------------------|----|
| Per million document tokens | ~$10-12 | **$1.02** | 90% cost reduction with caching |
| 10K documents (~50M tokens) | ~$500-600 | ~$51 | One-time processing cost |
| 100K documents (~500M tokens) | ~$5,000-6,000 | ~$510 | Significant but one-time |
| 1M documents (~5B tokens) | ~$50,000-60,000 | ~$5,100 | At scale, caching is essential |

**Prompt caching mechanism:** The full document is cached on the first chunk, then reused for all subsequent chunks from the same document. This reduces latency by >2x and cost by ~90%.

### 6.4 Benchmark Results

Measured as retrieval failure rate (1 - Recall@20):

| Strategy | Failure Rate | Improvement vs Baseline |
|----------|-------------|------------------------|
| Standard embeddings (baseline) | 5.7% | -- |
| Contextual Embeddings alone | 3.7% | 35% reduction |
| Contextual Embeddings + Contextual BM25 | 2.9% | 49% reduction |
| Contextual Embeddings + Contextual BM25 + Reranking | **1.9%** | **67% reduction** |

**Tested across domains:** Codebases, fiction, ArXiv papers, science papers.

**Top embedding models tested:** Gemini Text 004, Voyage AI models.

### 6.5 How Contextual BM25 Differs from Standard BM25

Standard BM25 indexes the raw chunk text. Contextual BM25 indexes the contextualized chunk (context prepended to original text). This means:
- BM25 can now match terms that appear in the document-level context but not in the chunk itself
- Entity names, section titles, and document identifiers become searchable within each chunk
- The chunk "The company's revenue grew by 3%" can now be found by searching "ACME" or "Q2 2023"

### 6.6 Integration with Reranking

The recommended full pipeline:
1. **Initial retrieval:** Top 150 chunks via contextual embeddings + contextual BM25 (hybrid)
2. **Reranking:** Cross-encoder model (Cohere, BGE, etc.) scores all 150 chunks against the query
3. **Final selection:** Top 20 chunks passed to the generative model

This three-stage pipeline achieves the maximum 67% failure reduction.

### 6.7 Domain-Specific Prompt Variations

For different document types, customize the prompt:
- **Code:** "Describe the module, class, and function this code belongs to"
- **Legal:** "Identify the contract section, parties, and clause type"
- **Scientific:** "State the paper title, section, and what hypothesis or result is being discussed"

---

## 7. Multi-Vector Retrieval (ColBERT)

### 7.1 ColBERT Architecture

ColBERT (Contextualized Late Interaction over BERT) is a multi-vector retrieval model based on BERT-base (~110M parameters).

**Key differences from bi-encoders:**
- Bi-encoders: pool all token embeddings into ONE vector per document
- ColBERT: keeps per-token embeddings (128-dimensional via projection from BERT's 768-dim)
- Special tokens `[Q]` and `[D]` distinguish queries from documents
- Documents are pre-encoded offline; queries are encoded at search time

### 7.2 MaxSim Scoring Algorithm

```
Score(Q, D) = SUM over qi in Q of: MAX over dj in D of: cosine_sim(qi, dj)
```

**Step-by-step:**
1. For each query token qi, compute cosine similarity with ALL document tokens
2. Keep only the MAXIMUM similarity for each query token
3. Sum all per-query-token maximum scores

This "late interaction" preserves token-level semantics while remaining scalable (document encodings are pre-computed).

### 7.3 Storage Requirements

ColBERT stores one 128-dimensional vector per token, creating substantial overhead:

| Metric | Single-Vector | ColBERT (100 tokens/doc) |
|--------|--------------|--------------------------|
| Vectors per 1K docs | 1,000 | 100,000 |
| Vectors per 1M docs | 1,000,000 | 100,000,000 |
| Storage per 1M docs (uncompressed) | ~512 MB | ~51.2 GB |

**ColBERTv2 compression (residual compression):**
- Combines high-precision centroid vectors with low-precision residual vectors
- MS MARCO index: 154 GB (v1) reduced to 16-25 GB (v2) -- 6-10x reduction
- 1-bit compression: 16 GB; 2-bit compression: 25 GB

### 7.4 RAGatouille Library

RAGatouille wraps ColBERT with simple APIs:

```python
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
RAG.index(
    collection=["doc1 text", "doc2 text", ...],
    index_name="my_index",
)

# Search
results = RAG.search(query="my question", k=10)
```

**Features:** Indexing, search, fine-tuning/training on custom data.

### 7.5 Performance Benchmarks

- ColBERT consistently outperforms single-vector bi-encoders on complex queries requiring token-level matching
- Particularly strong for: legal document analysis, financial verification, multi-aspect queries
- Explainability advantage: can show which specific tokens matched between query and result
- Trade-off: 10-100x more storage than single-vector, slower indexing

---

## 8. Ensemble Retrieval

### 8.1 Core Concept

Ensemble retrieval combines results from multiple independent retrievers, each potentially using different strategies, embeddings, or configurations.

### 8.2 LangChain EnsembleRetriever

```python
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever, splade_retriever],
    weights=[0.4, 0.3, 0.3],  # Defaults to equal weights if omitted
    c=60,  # RRF smoothing constant
    id_key="doc_id",  # Metadata key for deduplication (default: page_content)
)

results = ensemble_retriever.invoke("my query")
```

**Algorithm: Weighted Reciprocal Rank Fusion**
```
score(d) = SUM over i of: weight_i / (c + rank_i(d))
```

1. Each retriever produces a ranked list
2. For each document in each list, calculate weighted RRF score
3. Duplicate documents (same page_content or id_key) accumulate scores
4. Sort by final score descending

### 8.3 When to Use Ensemble vs Single Hybrid

| Use Case | Ensemble (Multiple Retrievers) | Single Hybrid (Dense + Sparse) |
|----------|-------------------------------|-------------------------------|
| Multiple embedding models (e.g., code + text) | Yes | No |
| Different index strategies per document type | Yes | No |
| Standard text search | Overkill | Yes |
| Maximum recall at cost of latency | Yes | Usually sufficient |
| Simple deployment | No (more infrastructure) | Yes |

### 8.4 Weighting Strategies

- **Equal weights:** Start here as baseline
- **Task-specific:** Weight retrievers based on evaluation on your domain
- **Query-dependent:** Route different query types to different weight distributions (requires a classifier)
- **Learned weights:** Train a small model to predict optimal weights per query (advanced)

---

## 9. Retrieval Evaluation Metrics

### 9.1 Core Metrics Reference

| Metric | Formula | Rank-Aware? | Binary/Graded | When to Use |
|--------|---------|-------------|---------------|-------------|
| **Precision@K** | (Relevant in top K) / K | No | Binary | "How many of my top K results are useful?" |
| **Recall@K** | (Relevant in top K) / (Total relevant) | No | Binary | "What fraction of all relevant docs did I find?" |
| **MRR** | (1/\|Q\|) * SUM(1 / rank of first relevant) | Yes | Binary | "How quickly do I find the first correct answer?" |
| **MAP@K** | (1/\|Q\|) * SUM(Average Precision per query) | Yes | Binary | "Are relevant results consistently ranked high?" |
| **NDCG@K** | DCG@K / IDCG@K | Yes | **Graded** | "Are results sorted by degree of relevance?" |

### 9.2 Detailed Formulas

**Precision@K:**
```
Precision@K = |{relevant docs in top K}| / K
```

**Recall@K:**
```
Recall@K = |{relevant docs in top K}| / |{all relevant docs}|
```

**Mean Reciprocal Rank (MRR):**
```
MRR = (1/|Q|) * SUM over q in Q of: 1 / rank_q(first relevant result)
```
Only considers the first relevant result per query. Range: [0, 1]. MRR=1 means the first result is always correct.

**Average Precision (AP@K) for a single query:**
```
AP@K = (1/N) * SUM over k=1..K of: Precision@k * rel(k)
```
Where N = total relevant docs, rel(k) = 1 if result at rank k is relevant, 0 otherwise.

**MAP@K (Mean Average Precision):**
```
MAP@K = (1/|Q|) * SUM over q in Q of: AP@K(q)
```

**NDCG@K:**
```
DCG@K = SUM over i=1..K of: rel_i / log2(i + 1)
IDCG@K = DCG@K of the ideal (perfect) ranking
NDCG@K = DCG@K / IDCG@K
```
Unique advantage: supports graded relevance (not just binary). Default metric for MTEB Retrieval leaderboard.

### 9.3 Which Metric for RAG?

| RAG Scenario | Primary Metric | Rationale |
|-------------|----------------|-----------|
| Single-answer Q&A | MRR | Only need the first correct chunk |
| Multi-chunk synthesis | Recall@K | Need to find all relevant chunks |
| Ranked passage display | NDCG@K | Order matters for user experience |
| General retrieval quality | MAP@K | Balances precision and ranking |
| Production monitoring | Recall@20 + Precision@5 | Recall ensures coverage; precision reduces noise |

### 9.4 Evaluating Retrieval Quality Independently

**Why evaluate retrieval separately from generation:**
- Isolates retrieval failures from generation failures
- Cheaper to iterate (no LLM calls needed)
- More reproducible (deterministic retrieval vs. stochastic generation)

**Methodology:**
1. Create a labeled evaluation set: 50-200 query-relevant_passage pairs
2. Run queries through your retrieval pipeline
3. Compute Recall@K (where K = your top_k setting)
4. Target: Recall@20 > 0.95 for production systems
5. Iterate on chunking, embedding model, and retrieval strategy

---

## 10. Production Considerations

### 10.1 Retrieval Latency Budgets

**Target: p95 ~1.2s to first useful tokens (FTT)**

| Component | P50 Target | P95 Target | Notes |
|-----------|-----------|-----------|-------|
| Query Understanding | 30-60 ms | 60-120 ms | Normalize, intent detection, entity expansion |
| Embedding Generation | 20-50 ms | 50-100 ms | GPU batch inference; 16-64 queries per batch |
| ANN Vector Search | 5-20 ms | 20-50 ms | HNSW on hot data in RAM |
| Sparse/BM25 Search | 5-15 ms | 15-40 ms | Inverted index, typically faster than ANN |
| Score Fusion (RRF/weighted) | 1-5 ms | 5-10 ms | Trivial computation |
| Reranking | 40-90 ms | 90-200 ms | Cross-encoder model. MiniLM-L-12 or BGE-base recommended for latency |
| Context Assembly | 20-40 ms | 40-80 ms | Deduplication, compression, formatting |
| Safety/PII Checks | 10-30 ms | 20-60 ms | Classifier-based, not LLM |
| LLM Generation | 150-300 ms | 250-450 ms | Dominates total latency. Streaming masks perceived wait |

**Scaling benchmarks:**
- Single node, 512 GB RAM: ~40M embeddings, sub-20ms p95 retrieval
- Hot data (last 30 days, ~10M embeddings): 5-10ms p95
- Cold data (disk-backed): 30-50ms p95
- Serverless vector DB: 50-100ms p95 single-region

### 10.2 top_k Selection and Generation Quality

**The top_k dilemma:** Too few chunks = missed information (low recall). Too many chunks = diluted context, higher hallucination risk, more latency and cost.

**Recommended approach (from Anthropic research):**
1. Initial retrieval: top 150 candidates (broad recall)
2. Reranking: score all 150 with cross-encoder
3. Final context: top 20 chunks to the LLM

**Practical guidelines:**

| top_k | Typical Use | Notes |
|-------|-------------|-------|
| 3-5 | Simple factoid Q&A | Fast, low cost, sufficient for single-answer queries |
| 10-20 | General RAG | Standard for most production systems |
| 20-50 | Complex analysis, multi-hop | Needs reranking to filter noise |
| 50-150 | Initial retrieval before reranking | Never send this many directly to LLM |

**Adaptive-k:** Rather than fixed top_k, dynamically select chunks based on score thresholds or score gaps. Research shows a "buffer" of B=5 additional chunks beyond the adaptive cutoff helps.

**Key finding:** Chunking configuration has as much or more influence on retrieval quality as the choice of embedding model (Vectara/NAACL 2025).

### 10.3 Caching Strategies

| Cache Type | What It Stores | TTL | Impact |
|------------|---------------|-----|--------|
| **Embedding Cache** | Query text -> embedding vector | 1 hour | Avoids redundant GPU inference for repeated/similar queries |
| **Result Cache** | Query embedding -> top_k results | 30 minutes | Skips vector search entirely for cached queries |
| **Rerank Cache** | (query, passage set) -> reranked order | 30 minutes | Avoids cross-encoder inference |
| **Response Cache** | Full query -> LLM response | 5-15 minutes | Eliminates entire pipeline for exact repeats |
| **Semantic Cache** | Embedding similarity match -> cached response | Varies | Reuses answers for semantically similar (not identical) queries. Risk: false positives |
| **KV-Cache** | Document key-value tensors | Session/TTL | Reuses LLM attention computations across requests sharing documents |

**Production pattern:** 2% of documents account for ~60% of retrieval requests. Caching these "hot" documents' embeddings and retrieval results yields outsized efficiency gains.

**Semantic cache warning:** False positive matches (returning cached answers for genuinely different questions) are a real risk, especially in domains with subtle distinctions. A banking case study showed this requires careful threshold tuning.

### 10.4 Approximate vs Exact Nearest Neighbor

| Aspect | Exact (Brute Force) | HNSW | IVF | IVFPQ |
|--------|-------------------|------|-----|-------|
| **Recall** | 100% | 95-99.9% | 85-99% | 80-95% |
| **Latency (1M vectors)** | 50-500ms | 1-10ms | 2-20ms | 1-10ms |
| **Memory (1M x 768d)** | ~3 GB | ~4-6 GB | ~3 GB | ~0.1-0.5 GB |
| **Build time** | None | Minutes-hours | Minutes | Minutes |
| **Max practical scale** | ~100K vectors | ~10-50M vectors | ~100M vectors | ~1B+ vectors |
| **Best for** | Small datasets, evaluation | General production | Large scale | Billion-scale, memory-constrained |

**Decision framework:**
- < 100K vectors: exact search is fine
- 100K - 10M vectors: HNSW (standard choice)
- 10M - 100M vectors: HNSW with quantization, or IVF
- 100M+ vectors: IVFPQ or disk-based solutions

---

## Summary: Strategy Selection Guide

| Strategy | Best For | Complexity | Latency Impact |
|----------|----------|------------|----------------|
| **Dense only** | Semantic similarity, conversational queries | Low | Low |
| **BM25 only** | Exact matching, keyword-heavy domains | Low | Very low |
| **Hybrid (Dense + BM25)** | General-purpose RAG (recommended default) | Medium | Low-medium |
| **Hybrid + Reranking** | Production RAG needing high accuracy | Medium-high | Medium |
| **Contextual Retrieval** | Maximum retrieval quality, willing to pay preprocessing cost | High | Low (after preprocessing) |
| **Small-to-Big** | Documents with filler text diluting embeddings | Medium | Low |
| **Sentence Window** | Dense documents with varied topics per paragraph | Medium | Low |
| **ColBERT (Multi-Vector)** | Complex queries needing token-level matching | High | Medium |
| **Ensemble (3+ retrievers)** | Multi-modal or multi-domain corpora | Very high | High |
| **Contextual + Hybrid + Reranking** | Maximum quality (Anthropic's recommended stack) | Very high | Medium |

---

## Key Sources

- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Elastic: Practical BM25 Part 2](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [Elastic: Practical BM25 Part 3](https://www.elastic.co/blog/practical-bm25-part-3-considerations-for-picking-b-and-k1-in-elasticsearch)
- [Pinecone: HNSW Explained](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Pinecone: SPLADE Explained](https://www.pinecone.io/learn/splade/)
- [Weaviate: Late Interaction Overview (ColBERT)](https://weaviate.io/blog/late-interaction-overview)
- [Weaviate: Retrieval Evaluation Metrics](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [Qdrant: Modern Sparse Neural Retrieval](https://qdrant.tech/articles/modern-sparse-neural-retrieval/)
- [OpenSearch: HNSW Hyperparameter Guide](https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/)
- [LangChain: ParentDocumentRetriever](https://python.langchain.com/docs/how_to/parent_document_retriever/)
- [LlamaIndex: Recursive Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/)
- [LlamaIndex: SentenceWindowNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/)
- [LangChain: EnsembleRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html)
- [Superlinked: Optimizing RAG with Hybrid Search](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Milvus: IVF vs HNSW](https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md)
- [RAG Latency Budgets](https://medium.com/@bhagyarana80/10-rag-latency-budgets-where-to-spend-your-milliseconds-5733f6483316)
- [SPLADE GitHub (Naver)](https://github.com/naver/splade)
- [ColBERT GitHub (Stanford)](https://github.com/stanford-futuredata/ColBERT)
- [RAGatouille: ColBERT Made Easy](https://til.simonwillison.net/llms/colbert-ragatouille)
- [Dense Retrieval Failure Modes](https://chamomile.ai/challenges-dense-retrieval/)
- [Assembled: RRF and Hybrid Search](https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search)
- [OpenSearch: RRF for Hybrid Search](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [BM25 Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [SPLADEv2 Paper](https://arxiv.org/abs/2109.10086)
- [Adaptive-k for RAG](https://datasciocean.com/en/paper-intro/adaptive-k/)
- [RAGCache: Efficient Knowledge Caching](https://arxiv.org/pdf/2404.12457)
