# Post-Retrieval Optimization & Advanced Generation Techniques in RAG Systems

**Research Date:** 2026-04-04
**Scope:** Re-ranking models, context compression, advanced generation techniques

---

## Part 1: Re-ranking Models (Deep Dive)

### 1.1 Cross-Encoders

#### How Cross-Encoders Work

Cross-encoders jointly encode query-document pairs through a single transformer pass. Unlike bi-encoders (which encode query and document separately), cross-encoders concatenate the query and document as a single input `[CLS] query [SEP] document [SEP]` and produce a single relevance score. This joint attention across both texts enables full token-level interaction, achieving higher accuracy at the cost of computational efficiency.

**Key trade-off:** Cross-encoders cannot pre-compute document embeddings -- every query requires a fresh forward pass for each candidate document, making them O(n) per query vs O(1) for bi-encoders at retrieval time. This is why they are used as **re-rankers** on a small candidate set (typically 20-100 documents) rather than as first-stage retrievers.

#### ms-marco-MiniLM-L-12-v2

| Property | Value |
|----------|-------|
| Architecture | MiniLM (12 layers, distilled from BERT-large) |
| Parameters | ~33.4M |
| Training Data | MS MARCO passage ranking (530K queries, 8.8M passages) |
| Latency (100 docs, CPU) | ~50ms |
| Latency (100 docs, GPU) | ~15-20ms |
| BEIR NDCG@10 (avg) | ~55.43 (FlashRank variant) |

- Remains the best lightweight open-source option for English-only workloads
- Achieves +35% accuracy improvement over embedding-only retrieval at 50ms for 100 document pairs
- Runs effectively on CPU without GPU requirements

#### bge-reranker-large and bge-reranker-v2-m3

| Property | bge-reranker-large | bge-reranker-v2-m3 |
|----------|-------------------|-------------------|
| Parameters | ~560M | ~568M |
| Languages | English-focused | 100+ languages |
| Latency (3 docs, CPU) | ~350ms | ~350ms |
| Latency (3 docs, T4 GPU) | ~80ms | ~80ms |
| Latency (100 docs, GPU) | 50-100ms | 50-100ms |
| BEIR NDCG@10 | Strong | Among top open-source |

- bge-reranker-v2-m3 is the recommended multilingual production model
- After BGE reranking: Precision@10 improves from 0.62 to 0.84 (+22%), Top-3 recall from 0.45 to 0.79

#### BAAI/bge-reranker-v2-gemma

- Based on Google Gemma architecture (larger model)
- Latest generation reranker from BAAI
- Improved reasoning capabilities for complex queries
- Higher computational requirements than MiniLM-based models

#### NV-RerankQA-Mistral-4B-v3

- Based on Mistral 7B adapted as cross-encoder
- Provides the highest ranking accuracy across all datasets by a large margin (+14% compared to second-best bge-reranker-v2-m3)
- Demonstrates the effectiveness of adapting large LMs as cross-encoders

#### Latency and Batch Processing Optimization

| Model | 3 docs (CPU) | 3 docs (GPU T4) | 100 docs (CPU) | 100 docs (GPU) |
|-------|-------------|-----------------|----------------|----------------|
| MiniLM-L-6-v2 | ~15ms | ~5ms | ~50ms | ~15ms |
| MiniLM-L-12-v2 | ~25ms | ~8ms | ~80ms | ~25ms |
| BGE-reranker-v2-m3 | ~350ms | ~80ms | ~1.2s | 50-100ms |
| NV-Mistral-4B | N/A | ~200ms | N/A | ~500ms |

**Batch processing tips:**
- Use dynamic batching to maximize GPU utilization
- Sort documents by length to minimize padding waste
- Use ONNX Runtime or TensorRT for 2-3x inference speedup
- For >1000 queries/sec, GPU inference is required (AWS g4dn.xlarge handles ~2000 pairs/second)

#### GPU Requirements for Production

| Scale | Recommended GPU | Notes |
|-------|----------------|-------|
| Prototype/Dev | CPU (any) | MiniLM models run fine |
| Small production (<100 QPS) | T4 (16GB) | Sufficient for BGE models |
| Medium production (100-1000 QPS) | A10G (24GB) | Good price/performance |
| High production (>1000 QPS) | A100 (40/80GB) | For Mistral-based rerankers |

#### Fine-tuning Cross-Encoders on Custom Data

**Training format:** `(query, document, relevance_label)` triples

**Using sentence-transformers v3:**
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Training data: list of InputExample(texts=[query, doc], label=score)
model.fit(train_dataloader, epochs=2, warmup_steps=100)
```

**Training requirements:**
- As few as ~5,000 labeled pairs can yield significant improvements
- Fine-tuning takes 1-2 hours on a modest GPU
- With 4,719 QA pairs: ~1 minute on NVIDIA A10G, cost < $0.10
- CPU training works but takes 3-4x longer
- CUDA-compatible GPU recommended for larger datasets

**Data preparation strategies:**
- Use hard negatives from your retriever (BM25 or dense)
- Label schema: binary (relevant/not) or graded (0-3)
- Include domain-specific query patterns

---

### 1.2 Cohere Rerank

#### Rerank 3.5 vs Rerank 3 vs Rerank 3 Nimble

| Feature | Rerank 3 | Rerank 3 Nimble | Rerank 3.5 |
|---------|----------|-----------------|------------|
| Context Length | 4096 tokens | 4096 tokens | 4096 tokens |
| Speed | Baseline | 3-5x faster than Rerank 3 | Improved over 3.0 |
| Multilingual | 100+ languages | 100+ languages | 100+ languages |
| Reasoning | Standard | Standard | Enhanced constraint understanding |
| Latency (measured) | ~150ms | ~40-60ms | ~130ms (100-150ms range) |
| Key Strength | Broad capability | Speed-sensitive apps | Complex enterprise queries |

**Rerank 3.5 improvements over 3.0:**
- Better performance on constrained queries (e.g., "find contracts from 2023 only")
- SOTA performance on BEIR and domain-specific benchmarks (Finance, E-commerce, Hospitality, Project Management, Email/Messaging)
- Improved handling of nuanced relevance

#### API Details

**Input format:**
```json
{
  "model": "rerank-v3.5",
  "query": "user query",
  "documents": ["doc1", "doc2", ...],
  "top_n": 5,
  "return_documents": true
}
```

**Limits:**
- Max documents per request: 1,000
- Max tokens per document: 4,096 (including query)
- If a document exceeds 500 tokens (including query), it is automatically split into multiple chunks
- Each chunk counts as a separate search unit for billing

#### Pricing Model

| Tier | Cost |
|------|------|
| Per 1,000 searches | $2.00 |
| Search unit definition | 1 query + up to 100 documents |
| Document >500 tokens | Auto-split into multiple chunks (each counts separately) |
| Free tier | Available for development |

**Cost comparison for 100K queries/month:**
- Cohere Rerank 3.5: ~$200
- Open-source (self-hosted GPU): $0 (infrastructure cost only, ~$150-300/month for GPU)
- Open-source (CPU only): $0 (minimal infrastructure)

#### Multilingual Performance

- Trained for 100+ languages
- SOTA on multilingual retrieval benchmarks
- Particularly strong in: English, Chinese, Japanese, Korean, European languages
- Handles cross-lingual queries (query in English, documents in other languages)

#### LangChain Integration

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

cohere_rerank = CohereRerank(
    cohere_api_key="YOUR_KEY",
    model="rerank-v3.5",
    top_n=5
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_rerank,
    base_retriever=base_retriever
)
```

#### LlamaIndex Integration

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(
    api_key="YOUR_KEY",
    model="rerank-v3.5",
    top_n=5
)
query_engine = index.as_query_engine(
    node_postprocessors=[cohere_rerank]
)
```

#### When Cohere Beats Open-Source

1. **Multilingual workloads**: Superior cross-lingual performance out-of-box
2. **No GPU budget**: Avoids GPU infrastructure costs
3. **Rapid prototyping**: Zero setup, immediate results
4. **Complex constraint queries**: Rerank 3.5 excels at nuanced relevance
5. **Low volume (<50K queries/month)**: Cost-effective vs GPU provisioning
6. **Enterprise compliance**: Available on AWS Bedrock, Oracle Cloud, SageMaker

---

### 1.3 ColBERT (Late Interaction)

#### Architecture: Per-Token Embeddings and MaxSim

ColBERT uses a **late interaction** paradigm:

1. **Encoding:** Each query and document is independently encoded by BERT into per-token embeddings, then projected to lower dimensionality (768 -> 128d) via a linear layer
2. **Query encoding:** `E_q = Normalize(CNN(BERT("[Q] q_1 q_2 ... q_n [mask]...")))`
3. **Document encoding:** `E_d = Normalize(CNN(BERT("[D] d_1 d_2 ... d_m")))`

**MaxSim Operation (Mathematical Formula):**

```
S(q, d) = Sum_{i=1}^{|q|} max_{j=1}^{|d|} (E_q_i . E_d_j^T)
```

For each query token embedding, find the maximum cosine similarity with any document token embedding, then sum these maximums across all query tokens. This allows fine-grained token-level matching while keeping document encodings pre-computable.

#### ColBERTv1 vs ColBERTv2

| Feature | ColBERTv1 | ColBERTv2 |
|---------|-----------|-----------|
| Compression | None (full FP32 vectors) | Residual compression (6-10x reduction) |
| Supervision | Standard negatives | Denoised supervision (distillation from cross-encoder) |
| Storage per 1M docs | ~25-50 GB | ~3-8 GB |
| Quality (MS MARCO MRR@10) | 36.0 | 39.7 |
| Quality (BEIR NDCG@10 avg) | ~49.x | ~54.20 |
| Indexing | Full vectors | Centroids + residuals |

**ColBERTv2 key innovations:**
- **Residual compression:** Clusters token embeddings into centroids, stores only residuals (differences from nearest centroid), enabling 6-10x storage reduction
- **Denoised supervision:** Uses a cross-encoder teacher to generate soft labels, reducing noise from hard negative sampling

#### RAGatouille Library

```python
from ragatouille import RAGPretrainedModel

# Load pre-trained ColBERT
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
RAG.index(
    collection=["doc1 text", "doc2 text", ...],
    index_name="my_index",
    split_documents=True,  # auto-chunking
    max_document_length=256
)

# Search
results = RAG.search(query="your query", k=10)

# Fine-tune on custom data
RAG.train(
    training_data=[{"query": "q", "positive": "pos_doc", "negative": "neg_doc"}],
    nbits=2,  # compression level
)
```

#### PLAID Engine

PLAID (Performance-optimized Late Interaction Driver) accelerates ColBERT search:

- **Centroid interaction:** Treats each passage as a lightweight bag of centroids for fast initial filtering
- **Speed improvements:**
  - 7x faster on GPU vs vanilla ColBERTv2
  - 45x faster on CPU vs vanilla ColBERTv2
- **Three-stage pipeline:** centroid pruning -> candidate generation -> full late interaction on top candidates

#### Storage Requirements

| Configuration | Storage per 1M docs (128 tokens avg) | Monthly cost (100M docs, AWS) |
|--------------|--------------------------------------|------------------------------|
| 128d, no compression | ~25 GB | ~$1,319/month |
| 128d, 2-bit quantization | ~6 GB | ~$330/month |
| 64d, no compression | ~12.5 GB | ~$660/month |
| 64d, 2-bit quantization | ~3 GB | ~$165/month |
| 128d + pool factor 2 + 2-bit | ~3 GB | ~$165/month |

**Per-token storage:** Each token = 128 floats x 4 bytes = 512 bytes (uncompressed). With 2-bit quantization: ~32 bytes per token.

#### Matryoshka ColBERT (Variable Dimensionality)

Jina ColBERT v2 supports Matryoshka Representation Learning:

| Dimensionality | Relative Performance | Storage Savings |
|---------------|---------------------|-----------------|
| 128d (default) | 100% (baseline) | 0% |
| 96d | ~99% (~1% drop) | 25% |
| 64d | ~98.5% (~1.5% drop) | 50% |

#### Token Pooling (Answer.AI Research)

| Pool Factor | Vector Count Reduction | Performance (no quantization) | Performance (with 2-bit quant) |
|-------------|----------------------|------------------------------|-------------------------------|
| 2 | 50% | 100.6% (slight improvement) | <3% drop |
| 3 | 66% | 99% | ~4% drop |
| 4 | 75% | 97% | ~6% drop |

Combined with 2-bit quantization, pool factor 2 achieves ~16.7% of original storage with minimal quality loss. No model retraining required -- works out-of-the-box.

#### Benchmarks: MS MARCO and BEIR

| Model | MS MARCO MRR@10 | BEIR NDCG@10 (avg) |
|-------|----------------|-------------------|
| ColBERTv2 | 39.7 | 54.20 |
| Jina-ColBERT-v2 | ~40.x | ~55.x |
| BM25 (baseline) | 18.7 | 44.x |
| Dense retriever (e5-large) | 38.x | 50.x |

---

### 1.4 FlashRank

#### Architecture and Model Variants

FlashRank is an ultra-lightweight Python library for reranking using ONNX Runtime with CPU/GPU execution and INT8 quantized weights.

| Model Variant | Size | Performance Profile | Use Case |
|--------------|------|-------------------|----------|
| ms-marco-TinyBERT-L-2-v2 (default) | ~4 MB | Blazing fast, competitive | High-throughput, latency-critical |
| ms-marco-MiniLM-L-12-v2 | ~34 MB | Best cross-encoder quality | Quality-first applications |
| rank-T5-flan | ~110 MB | Best zero-shot OOD | Out-of-domain generalization |
| ms-marco-MultiBERT-L-12 | ~150 MB | Multilingual (100+ lang) | International applications |
| ce-esci-MiniLM-L12-v2 | ~34 MB | E-commerce optimized | Product search |
| rank_zephyr_7b_v1_full | ~4 GB | 4-bit GGUF, LLM-based | Maximum quality listwise |

#### CPU-Only Performance

- Sub-20ms reranking of 50 candidates on CPU
- No PyTorch or Transformers dependency required
- ONNX Runtime with hardware-specific optimizations
- Minimal memory footprint (4MB for default model)

```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=128)
rerankrequest = RerankRequest(query="query", passages=[
    {"id": 1, "text": "doc text", "meta": {"source": "wiki"}}
])
results = ranker.rerank(rerankrequest)
```

#### Comparison with Cross-Encoders

| Metric | Full Cross-Encoder | FlashRank |
|--------|-------------------|-----------|
| Accuracy (relative) | 100% (baseline) | 95-98% |
| Speed (relative) | 1x | 10-30x faster |
| GPU Required | Recommended | No |
| Dependencies | PyTorch, Transformers | ONNX Runtime only |
| BEIR NDCG@10 (TinyBERT) | N/A | ~52.x |
| BEIR NDCG@10 (MiniLM-12) | ~55.43 | ~55.43 |

#### When FlashRank is "Good Enough"

1. **Serverless/edge deployment:** 4MB model fits anywhere
2. **Budget constraints:** No GPU costs
3. **High throughput:** 10-30x faster than full cross-encoders
4. **General-domain queries:** Quality gap is 2-5% on standard benchmarks
5. **NOT recommended for:** Complex reasoning queries, domain-specific ranking, multilingual (use MultiBERT variant)

---

### 1.5 LLM-based Reranking

#### RankGPT: Listwise Reranking

RankGPT uses instruction-tuned LLMs to perform **listwise** reranking -- processing an entire candidate list in a single inference pass and outputting a complete permutation.

**Prompting approach:**
```
Given a query and a list of passages, rank the passages by relevance.
Query: {query}
Passages:
[1] {passage_1}
[2] {passage_2}
...
[20] {passage_20}
Output the ranking as a permutation of passage numbers.
```

**Sliding window strategy:** For >20 documents, RankGPT uses a sliding window approach, processing subsets and merging rankings.

#### RankLLM Framework

RankLLM is a comprehensive Python toolkit supporting multiple reranking paradigms:

| Mode | Models | Description |
|------|--------|-------------|
| Pointwise | MonoT5 | Scores each document independently |
| Pairwise | DuoT5 | Compares document pairs |
| Listwise | RankGPT, RankGemini, open-source LLMs | Ranks entire list at once |

**Supported backends:** vLLM, SGLang, TensorRT-LLM for open-source models; OpenAI, Google APIs for proprietary models.

**Prompt templates:** RankGPT (default), RankGPT-APEER, LRL (Learning to Rank with Language models).

**Zero-shot by default:** No in-context examples included, but supports configurable few-shot prompting.

#### Prompting Strategies for Reranking

1. **Listwise (RankGPT):** Present all candidates, ask for full permutation -- best accuracy
2. **Pointwise (MonoT5):** Score each document independently -- most parallelizable
3. **Pairwise (DuoT5):** Compare pairs and aggregate -- good for small candidate sets
4. **Setwise:** Score subsets and merge -- balance of quality and efficiency
5. **AFR-Rank:** Filter noise documents first, then rank -- 2.7x efficiency improvement over RankGPT

#### Cost Analysis vs Model-Based Rerankers

| Reranker | Cost per 1K queries (20 docs each) | Latency | Quality (BEIR) |
|----------|-----------------------------------|---------|----------------|
| GPT-4 (RankGPT) | ~$2-5 | 3-10s | Highest (NDCG@10 ~60.75 with MonoT5-3B) |
| GPT-3.5 (RankGPT) | ~$0.20-0.50 | 1-3s | Very high |
| Cohere Rerank 3.5 | $2.00 | 100-150ms | High |
| Cross-encoder (self-hosted) | ~$0.01-0.05 (GPU cost) | 50-100ms | High |
| FlashRank (CPU) | ~$0.001 | <20ms | Good |

#### When LLM Reranking is Worth the Cost

1. **Complex reasoning queries:** Where relevance requires deep understanding
2. **Zero-shot domain transfer:** LLMs generalize better than fine-tuned cross-encoders to new domains
3. **Low-volume, high-value:** Legal discovery, medical research (quality > cost)
4. **When you need explanations:** LLMs can explain why a document is relevant
5. **NOT worth it for:** High-throughput applications, simple factoid queries, budget-constrained systems

---

### 1.6 Reranking Benchmarks Summary

#### NDCG@10 on BEIR (Average Across Datasets)

| Model | NDCG@10 | Type |
|-------|---------|------|
| jina-reranker-v3 | 61.85 | Late interaction + cross-encoder |
| mxbai-rerank-large-v2 (1.5B) | 61.44 | Cross-encoder |
| MonoT5-3B-10k | 60.75 | Pointwise LLM |
| Twolar-xl | 60.03 | Cross-encoder |
| mxbai-rerank-large (Mixedbread) | 57.49 | Cross-encoder |
| mxbai-rerank-base (Mixedbread) | 55.57 | Cross-encoder |
| FlashRank (MiniLM-L-12) | 55.43 | Lightweight cross-encoder |
| ColBERTv2 | 54.20 | Late interaction |
| BM25 (baseline) | ~44.x | Lexical |

#### Latency Comparison

| Model | 20 docs | 100 docs | 1000 docs |
|-------|---------|----------|-----------|
| FlashRank (TinyBERT, CPU) | <5ms | <20ms | ~200ms |
| FlashRank (MiniLM-12, CPU) | ~10ms | ~45ms | ~400ms |
| Cross-encoder (MiniLM, GPU) | ~8ms | ~25ms | ~250ms |
| BGE-reranker-v2-m3 (GPU) | ~30ms | 50-100ms | ~500ms |
| Cohere Rerank 3.5 (API) | ~100ms | ~130ms | ~300ms |
| SPLADE (GPU) | ~50ms | ~490ms | ~2s |
| RankGPT (GPT-4) | ~3s | ~8s | N/A (context limit) |

#### Cost per 1,000 Reranking Operations (20 docs/query)

| Solution | Cost |
|----------|------|
| FlashRank (CPU) | ~$0.001 |
| Self-hosted cross-encoder (GPU) | ~$0.01-0.05 |
| Cohere Rerank 3.5 | $2.00 |
| GPT-3.5 (RankGPT) | ~$0.30 |
| GPT-4 (RankGPT) | ~$3.00 |

#### How top_n Selection Affects Generation Quality

- Reranking is NOT always a "win button" -- returns diminish and are context-dependent
- When retriever already returns high-quality candidates, reranking headroom is limited
  - Example: LitSearch benchmark -- dense retriever improved recall@5 by 24.8pp over BM25, but adding LLM reranker gave only 4.4pp further improvement
- Optimal top_n depends on query complexity and retriever quality
- General guidance: top_n=3-5 for generation, top_n=10-20 for the reranking candidate pool
- Cross-encoder rerankers deliver measurably lower hallucination rates in RAG applications
- Recent trend: move from static "rerank top-k always" to dynamically deciding how many documents to include (DynamicRAG)

---

## Part 2: Context Compression

### 2.1 LLMLingua / LongLLMLingua

#### LLMLingua: Token-Level Compression

**Algorithm (coarse-to-fine):**
1. **Budget Controller:** Allocates compression budget across prompt components (instructions, demonstrations, question) to maintain semantic integrity
2. **Iterative Token Compression:** Uses a small LM (GPT-2 or LLaMA-7B) to compute token-level perplexity. Tokens with low information content (high predictability) are removed iteratively
3. **Distribution Alignment:** Instruction-tuning based method aligns the compressor LM's distribution with the target LLM

**Compression ratios and quality:**

| Compression Ratio | Performance Drop | Notes |
|-------------------|-----------------|-------|
| 2x | <0.5% | Nearly lossless |
| 5x | <1% | Minimal impact |
| 10x | ~1% | Good for most tasks |
| 20x | ~1.5% | Max recommended; tested on Chain-of-Thought prompts |

**Key properties:**
- No training of the target LLM required
- Works with any LLM (GPT-4, GPT-3.5, Claude, Mistral)
- Uses compact models (GPT-2-small, LLaMA-7B) as compressor
- Achieves up to 20x compression with only 1.5 point performance drop

#### LongLLMLingua: RAG-Optimized Compression

Specifically designed for RAG scenarios with multiple retrieved documents:

- **Mitigates lost-in-the-middle effect** by reordering and compressing context
- **Performance boost:** Up to 21.4% improvement on NaturalQuestions with GPT-3.5 using only 1/4 of tokens
- **Key innovation:** Uses question-aware compression -- tokens relevant to the query are preserved preferentially
- **Document reordering:** Ranks documents by relevance before compression, placing most relevant at beginning/end

#### LLMLingua-2: Data Distillation Approach

| Feature | LLMLingua | LLMLingua-2 |
|---------|-----------|-------------|
| Architecture | GPT-2/LLaMA compressor | BERT-level encoder (data-distilled from GPT-4) |
| Task | Token classification (keep/remove) | Token classification (keep/remove) |
| Speed | Baseline | 3-6x faster |
| Out-of-domain | Good | Better |
| Max compression | 20x | 14x (with similar quality) |
| Training | Unsupervised perplexity | Supervised (GPT-4 distillation) |

#### When to Use vs Not Use

**Use LLMLingua when:**
- Context window is nearly full and you need to fit more documents
- Reducing API costs is a priority (fewer tokens = lower cost)
- Latency matters (fewer tokens = faster generation)
- Working with long prompts (Chain-of-Thought, few-shot examples)

**Do NOT use when:**
- Context window has plenty of room
- Every token carries critical information (legal documents, code)
- Compression model adds unacceptable latency to pipeline
- Working with highly structured content (tables, JSON)

---

### 2.2 Extractive Compression

#### EXIT: Context-Aware Extractive Compression (ACL 2025)

**Three-stage algorithm:**
1. **Sentence Decomposition:** Retrieved documents split into individual sentences
2. **Context-Aware Classification:** Each sentence undergoes binary classification ("Yes"/"No") considering full document context. Uses lightweight single-token prediction (parallelizable)
3. **Document Reassembly:** Sentences scoring above threshold (tau=0.5) are recombined in original order

**Compression results:**
- Token retention: ~31.2% (68.8% reduction)
- At k=30 documents: 4,497 tokens -> 594 tokens (86.8% reduction)
- Compression time: ~0.36 seconds

**QA Accuracy improvements (Llama-3.1-8B reader):**

| Dataset | Uncompressed (EM) | EXIT (EM) | Improvement |
|---------|-------------------|-----------|-------------|
| NQ | 34.6 | 35.9 | +1.3 |
| TriviaQA | 58.8 | 60.8 | +2.0 |
| HotpotQA | 28.1 | 30.6 | +2.5 |
| 2WikiMultihopQA | 16.1 | 24.2 | +8.1 |

With 70B reader: average +3.7 EM and +3.3 F1 improvement over uncompressed baseline.

**Latency:** 0.88s total (vs 1.03s uncompressed) -- 14.6% faster despite compression overhead.

#### RECOMP-Extractive

- Selects top-k sentences whose embeddings are most similar to query
- Simpler but less effective than EXIT on multi-hop tasks
- NQ: 34.6 EM vs EXIT's 35.9
- HotpotQA: 23.4 EM vs EXIT's 30.6 (significant gap on multi-hop)

#### Provence

- Trains a lightweight DeBERTa model for sentence-level relevance scoring
- Retains sentences exceeding a predefined threshold
- More flexible threshold tuning than RECOMP

#### Cross-Encoder Based Sentence Scoring

- Use cross-encoder to score each sentence against the query
- Sort sentences by score, keep top-k or above threshold
- Preserves semantic integrity better than token-level compression
- Typical safe removal: 40-70% of context without quality degradation for factoid QA

#### How Much Context Can You Remove Safely

| Task Type | Safe Compression | Notes |
|-----------|-----------------|-------|
| Factoid QA | 60-85% removal | Only need 1-2 key sentences |
| Multi-hop QA | 40-60% removal | Need to preserve reasoning chains |
| Summarization | 30-50% removal | Need broader coverage |
| Legal/Medical | 20-30% removal | High precision required |

---

### 2.3 Context Window Management

#### Lost-in-the-Middle Effect

**Core finding:** LLMs achieve highest accuracy when relevant information appears at the **beginning or end** of the input context. Performance degrades by **>30%** when critical information is in the middle.

**Root cause:** Rotary Position Embedding (RoPE) introduces a long-term decay effect causing models to prioritize tokens at sequence boundaries.

**U-shaped performance curve data:**
- Position 1 (beginning): ~75-85% accuracy
- Position 5-15 (middle): ~45-55% accuracy (30%+ degradation)
- Position 20 (end): ~70-80% accuracy
- Tested across 10, 20, and 30-document contexts

#### Optimal Context Ordering Strategies

1. **Relevance-based U-ordering:** Place highest-ranked documents at beginning and end, lowest in middle
2. **Reverse relevance in middle:** Most relevant -> beginning, second most -> end, rest in middle by descending relevance
3. **LongLLMLingua reordering:** Compress and reorder simultaneously based on question-aware relevance
4. **Ms-PoE (Multi-scale Positional Encoding):** Improves middle-position accuracy by 20-40%
5. **IN2 Training:** Teaches models position-invariant information extraction

#### How Many Documents to Include

**Research consensus:** Keep only the **most relevant 3-5 documents** in the prompt.

**Diminishing returns evidence:**
- 1-3 documents: Significant quality improvement per document added
- 4-5 documents: Marginal improvements
- 6-10 documents: Often flat or negative returns (noise overwhelms signal)
- 10+ documents: Consistently degrades performance on most tasks

**Reranking mitigates but doesn't eliminate this:** Reranking can improve retrieval accuracy by 15-30%, but the diminishing returns curve still applies to the final context.

#### NVIDIA RULER Benchmark

**What it tests:** Synthetic tasks evaluating real context handling across 4 categories:
1. **Retrieval:** Finding specific information (needle-in-haystack variants)
2. **Multi-hop tracing:** Following chains of reasoning across context
3. **Aggregation:** Combining information from multiple locations
4. **Question answering:** Complex QA requiring full context understanding

**Key findings:**
- 13 official tasks, tested from 4K to 128K context
- **Effective context is typically 50-65% of advertised size**
- Only half of models claiming 32K+ context can effectively handle 32K (beating Llama-2-7B at 4K baseline of 85.6%)
- Models achieving near-perfect needle-in-haystack show large degradation on RULER tasks
- Accuracy consistently declines as context length increases
- GPT-4 was best-performing: highest at 4K, least degradation at 128K
- Existing benchmarks using random negatives don't capture challenges of hard negatives prevalent in real RAG applications

---

## Part 3: Advanced Generation Techniques

### 3.1 Self-RAG (Deep Dive)

#### Reflection Token Types

| Token | Purpose | Possible Values |
|-------|---------|----------------|
| **Retrieve** | Should we retrieve documents? | {yes, no, continue} |
| **IsRel** | Is the passage relevant to the query? | {relevant, irrelevant} |
| **IsSup** | Is the claim supported by the passage? | {fully supported, partially supported, no support} |
| **IsUse** | How useful is the overall response? | {5, 4, 3, 2, 1} |

#### Training Methodology

**Phase 1: Critic Model Training**
1. Collect 4K-20K supervised instances per token type using GPT-4 prompting
2. Initialize critic from Llama 2-7B
3. Train via conditional language modeling: maximize `log p_C(r|x,y)` where r = reflection token, x = input, y = output
4. Achieved >90% agreement with GPT-4 judgments on most categories

**Phase 2: Generator Model Training**
1. Create augmented corpus:
   - Evaluate retrieval necessity per segment using critic
   - Retrieve top-K passages for segments needing retrieval
   - Predict IsRel, IsSup tokens for each passage
   - Append IsUse scores
2. Expand generator vocabulary with reflection tokens
3. Mask retrieved text during loss calculation (only train on generation + reflection tokens)
4. Train with standard next-token objective: `maximize log p_M(y,r|x)`
5. Down-sample: discard 50% of instances without retrieval tokens (balancing)

**Dataset:** 150,000 curated instruction-output pairs from Open-Instruct and knowledge-intensive sources (LAMA, ALCE-ASQA, ARC)

#### Inference Algorithm (Step by Step)

```
Input: query x, retriever R, generator M
Output: response y

1. For each segment t:
   a. Predict Retrieve token given (x, y_{<t})
   b. IF Retrieve = "yes":
      i.   Retrieve K passages using R
      ii.  For each passage d_k:
           - Predict IsRel(d_k)
           - Generate output segment y_t conditioned on (x, d_k, y_{<t})
           - Predict IsSup(y_t, d_k) and IsUse(y_t)
      iii. Rank candidates using beam search with critique scores
      iv.  Select best (y_t, d_k) pair
   c. IF Retrieve = "no":
      i.   Generate y_t conditioned on (x, y_{<t})
      ii.  Predict IsUse(y_t)
   d. Append best y_t to response
2. Return complete response y
```

#### Beam Search Scoring Formula

```
f(y_t, d) = p(y_t | x, d, y_{<t}) + S(Critique)

S(Critique) = sum over G in {IsRel, IsSup, IsUse} of:
    w^G * s_t^G

where s_t^G = normalized probability of most-desirable token for group G
```

**Default weights:** IsRel = 1.0, IsSup = 1.0, IsUse = 0.5
**Beam width:** 2 (segment-level)
**Retrieval threshold:** 0.2 (default), 0 for citation-heavy tasks

#### Benchmark Results (6 Tasks)

| Task | Metric | Self-RAG 7B | Self-RAG 13B | Ret-LLaMA2-chat | ChatGPT | Ret-ChatGPT |
|------|--------|------------|-------------|-----------------|---------|-------------|
| PopQA | Accuracy | 54.9 | 55.8 | 51.8 | 29.3 | - |
| TriviaQA | Accuracy | 66.4 | 69.3 | 59.8 | 74.3 | - |
| PubHealth | Accuracy | 72.4 | 74.5 | 52.1 | 70.1 | - |
| ARC-Challenge | Accuracy | 67.3 | 73.1 | 37.9 | 75.3 | - |
| Biography | FactScore | 81.2 | 80.2 | 79.9 | 71.8 | - |
| ASQA | EM | 30.0 | 31.7 | 32.8 | 35.3 | - |

**Key findings:**
- Self-RAG 7B/13B matches or exceeds ChatGPT on 4/6 tasks despite 9-50x parameter disadvantage
- Citation precision: 70.3% vs 65.1% for Ret-ChatGPT
- Significant gains on knowledge-intensive tasks (PopQA, PubHealth)
- Competitive on open-ended generation (Biography FactScore: 81.2)

#### Fine-tuning Requirements

| Requirement | Details |
|------------|---------|
| Base model | Llama 2 (7B or 13B) |
| Training | Standard supervised learning (no RL/PPO) |
| Retriever | Contriever-MS MARCO (off-the-shelf, unchanged) |
| Dataset | 150K instruction-output pairs |
| Compute | Standard GPU training (similar to instruction-tuning) |
| Training overhead | Minimal vs standard fine-tuning (just expanded vocabulary) |

#### Can Self-RAG Be Approximated Through Prompting?

**Partial approximation is possible** but with significant limitations:

1. **Retrieval decision:** Can prompt LLM to assess whether retrieval is needed -- works reasonably
2. **Relevance judgment:** Can prompt for passage relevance -- works well
3. **Support verification:** Can prompt to check if generation is supported -- works reasonably
4. **Utility scoring:** Can prompt for quality assessment -- works well

**What you lose without fine-tuning:**
- No segment-level beam search (key to quality)
- No internalized reflection (each check requires separate LLM call = high latency and cost)
- No learned retrieval threshold (must use heuristics)
- Inference weights (w^G) are not optimized

**Practical approximation:** LangGraph CRAG implementation approximates Self-RAG behavior through explicit graph nodes for retrieval evaluation, generation, and hallucination checking, achieving similar patterns without fine-tuning.

---

### 3.2 CRAG (Corrective RAG) (Deep Dive)

#### T5-large Retrieval Evaluator

**Training methodology:**
- Model: T5-large (0.77B parameters)
- Training data: PopQA (14K samples, 1,399 reserved for test)
- Labels: +1 (positive/relevant), -1 (negative/irrelevant)
- Output: Relevance score on [-1, 1] scale per document
- Fine-tuned as a regression task

#### Decompose-Then-Recompose Algorithm

```
Input: Retrieved documents D = {d_1, ..., d_n}, query q
Output: Refined knowledge K

Step 1 - SEGMENT:
  For each document d_i:
    If len(d_i) <= 2 sentences: keep intact as single strip
    Else: split into multi-sentence strips (knowledge units)

Step 2 - FILTER:
  For each strip s_j:
    score_j = T5_evaluator(q, s_j)   # score in [-1, 1]
    If score_j < -0.5: discard strip
  Retain top-5 scored strips

Step 3 - RECOMPOSE:
  K = concatenate(selected strips, in original order)
  Return K as refined internal knowledge
```

#### Confidence Thresholds

| Action | Condition | What Happens |
|--------|-----------|-------------|
| **Correct** | At least one doc > upper threshold | Use decompose-recompose on internal knowledge |
| **Incorrect** | All docs < lower threshold | Discard all docs, trigger web search |
| **Ambiguous** | Between thresholds | Combine refined internal knowledge + web search |

**Dataset-specific thresholds:**

| Dataset | Upper Threshold | Lower Threshold |
|---------|----------------|-----------------|
| PopQA | 0.59 | -0.99 |
| PubHealth | 0.50 | -0.91 |
| ARC-Challenge | 0.50 | -0.91 |
| Biography | 0.95 | -0.91 |

#### Web Search Integration

When **Incorrect** action triggers:
1. ChatGPT rewrites the query into keyword-based search terms (mimicking search engine usage)
2. Google Search API retrieves top-5 URLs, prioritizing Wikipedia
3. Retrieved HTML pages are segmented by special tokens (`<<<p>>>`, `<<</p>>>`)
4. T5 evaluator scores each paragraph using same filtering as internal knowledge
5. Top relevant paragraphs concatenated as external knowledge

**Integration patterns:**
- **Tavily:** Real-time web search API, common in LangGraph implementations (`TavilySearchResults(k=3)`)
- **Serper:** Google Search API wrapper, fast and affordable
- **Google Search API:** Direct integration, highest coverage
- **DuckDuckGo:** Free alternative, privacy-focused

#### Benchmark Results

| Dataset | Metric | Standard RAG | CRAG (LLaMA2-7B) | Self-CRAG (SelfRAG-7B) | Improvement (CRAG vs RAG) |
|---------|--------|-------------|------------------|----------------------|--------------------------|
| PopQA | Accuracy | ~50.5% | 54.9% | 61.8% | +4.4% |
| Biography | FactScore | ~32.8 | 47.7 | 86.2 | +14.9 |
| PubHealth | Accuracy | ~22.9% | 59.5% | 74.8% | +36.6% |
| ARC-Challenge | Accuracy | ~38.3% | 53.7% | 67.2% | +15.4% |

**Correct action accuracy:** 78.1% (vs 51.4% vanilla RAG) -- a 26.7pp improvement.

#### LangGraph CRAG Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int

# Node functions
def retrieve(state): ...      # Vector DB retrieval
def grade_documents(state): ... # LLM grades relevance
def generate(state): ...       # LLM generates answer
def web_search(state): ...     # Tavily/Serper fallback
def rewrite_query(state): ...  # LLM rewrites query

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.add_node("rewrite", rewrite_query)

# Conditional edges
workflow.add_conditional_edges(
    "grade",
    decide_action,  # Returns "generate", "web_search", or "rewrite"
    {"generate": "generate", "web_search": "web_search", "rewrite": "rewrite"}
)
```

#### Self-CRAG Variant

Self-CRAG combines CRAG's retrieval correction with Self-RAG's self-reflection:
- CRAG improves **evidence quality** (corrects retrieval)
- Self-RAG improves **reasoning over evidence** (reflection tokens)
- Combined: CRAG cleans retrieval inputs, then Self-RAG reflects on and refines answers
- Self-CRAG consistently outperforms both standalone approaches across all benchmarks

---

### 3.3 FLARE (Forward-Looking Active Retrieval) (Deep Dive)

#### Token Probability Monitoring Algorithm

```
Input: query x, LLM M, retriever R, threshold theta
Output: response y

y = ""
while not complete:
    # Step 1: Generate temporary next sentence
    s_temp = M.generate_sentence(x, y)   # without retrieval

    # Step 2: Check token confidence
    low_conf_tokens = [t for t in s_temp if prob(t) < theta]

    # Step 3: Decision
    if len(low_conf_tokens) == 0:
        y += s_temp    # Accept sentence as-is
    else:
        # Step 4: Formulate retrieval query
        query = mask_low_confidence_tokens(s_temp)  # Implicit query
        # OR: query = generate_question(s_temp)      # Explicit query

        # Step 5: Retrieve and regenerate
        docs = R.retrieve(query)
        s_new = M.generate_sentence(x, y, docs)  # Conditioned on docs
        y += s_new

return y
```

**Threshold theta:** Typically 0.5-0.8 (higher = more frequent retrieval, better accuracy, higher latency)

#### Look-Ahead Sentence Generation

The key innovation: FLARE generates a **temporary future sentence** to anticipate what information will be needed, then uses this anticipation as a retrieval query. This is fundamentally different from:
- **Single retrieval RAG:** Retrieves once based on input query only
- **Fixed-interval RAG:** Retrieves every N tokens regardless of need
- **FLARE:** Retrieves only when the model is uncertain about what it's about to say

#### Retrieval Trigger Mechanism

Two approaches for query formulation when retrieval is triggered:

1. **Implicit Query (FLARE-Direct):**
   - Mask low-confidence tokens in the temporary sentence
   - Example: "Joe Biden attended [MASK] University" -> retrieves info about Biden's education
   - Simpler, works with any model that provides logprobs

2. **Explicit Query (FLARE-Instruct):**
   - Generate a natural language question from the uncertain sentence
   - Example: "Which university did Joe Biden attend?"
   - More effective but requires instruction-following capability

#### FLARE vs FLARE-Instruct

| Feature | FLARE-Direct | FLARE-Instruct |
|---------|-------------|----------------|
| Query method | Mask uncertain tokens | Generate explicit question |
| Model requirement | Any model with logprobs | Instruction-following model |
| Query quality | Good (depends on masking) | Better (natural questions) |
| Latency | Lower (no question generation) | Higher (extra generation step) |
| Complexity | Simpler | More complex |

#### LLM APIs That Provide Logprobs

| Provider | Logprobs Support | Notes |
|----------|-----------------|-------|
| OpenAI (Completions API) | Yes | Full logprobs, top-5 alternatives |
| OpenAI (Chat Completions) | Yes | Added in 2024 |
| Anthropic Claude | No | Does not expose logprobs |
| Open-source (vLLM, llama.cpp) | Yes | Full control over logprobs |
| Mistral API | Yes | Via AI/ML API |
| DeepInfra | Yes | For hosted open-source models |
| OpenRouter | ~23% of endpoints | Varies by provider and model |

**Implication:** FLARE is most naturally implemented with OpenAI or open-source models. Anthropic Claude requires alternative confidence estimation (e.g., verbalized confidence, sampling-based).

#### Benchmarks on Long-Form Generation

- Tested across 4 long-form knowledge-intensive tasks
- Achieves superior or competitive performance on all tasks
- Significant improvements in factual accuracy over standard RAG
- Most effective for tasks requiring multiple facts across different topics

#### Latency Analysis

| Metric | Value |
|--------|-------|
| Average retrieval triggers per response | 2-5 (depends on topic complexity) |
| Additional latency per retrieval | 200-500ms (retriever) + generation time |
| Total overhead vs single-retrieval RAG | 2-5x slower |
| Mitigation | Async retrieval, caching, faster retrievers |

**When FLARE is worth the latency:**
- Long-form generation (articles, reports, summaries)
- Topics spanning multiple knowledge domains
- When factual accuracy is paramount
- NOT worth it for: simple QA, conversational responses, real-time applications

---

### 3.4 Agentic RAG (Deep Dive)

#### Design Patterns Taxonomy

Based on the 2025 Agentic RAG survey (arxiv:2501.09136):

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Single Agent** | One LLM agent with tools for retrieval, generation, verification | Simple-to-moderate queries |
| **Multi-Agent** | Specialized agents collaborating (retriever, grader, generator, verifier) | Complex multi-step queries |
| **Reflection Loop** | Agent evaluates own output quality and iterates | Improving answer accuracy |
| **Planning** | Agent decomposes query before retrieving | Complex/compound questions |
| **Tool Use** | Agent dynamically selects tools (vector DB, web search, SQL, APIs) | Heterogeneous data sources |

#### Single Agent RAG: LangGraph Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from operator import add

class AgentState(TypedDict):
    question: str
    generation: str
    documents: Annotated[List[str], add]
    retry_count: int
    search_needed: bool

def route_query(state):
    """Router: decide if retrieval is needed"""
    # LLM analyzes query to determine routing
    ...

def retrieve(state):
    """Retriever: fetch from vector store"""
    ...

def grade_documents(state):
    """Grader: evaluate document relevance"""
    ...

def generate(state):
    """Generator: synthesize answer"""
    ...

def check_hallucination(state):
    """Hallucination checker: verify grounding"""
    ...

def rewrite_query(state):
    """Query rewriter: reformulate if needed"""
    ...

# Graph construction
graph = StateGraph(AgentState)
graph.add_node("route", route_query)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate)
graph.add_node("check", check_hallucination)
graph.add_node("rewrite", rewrite_query)

# Conditional edges create the agentic loop
graph.add_conditional_edges("grade", decide_to_generate,
    {"generate": "generate", "rewrite": "rewrite"})
graph.add_conditional_edges("check", verify_answer,
    {"accept": END, "retry": "rewrite"})
```

#### Multi-Agent RAG Patterns

**CrewAI pattern:**
```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Research Analyst",
    tools=[vector_search, web_search])
verifier = Agent(role="Fact Checker",
    tools=[cross_reference_tool])
writer = Agent(role="Response Writer",
    tools=[citation_formatter])

crew = Crew(agents=[researcher, verifier, writer],
    tasks=[research_task, verify_task, write_task],
    process=Process.sequential)
```

**AutoGen pattern:** Multi-agent conversation where agents discuss and refine answers.

#### Reflection Loop

```
Agent generates answer
  -> Agent evaluates: "Is this answer well-supported by retrieved evidence?"
  -> If no: Agent rewrites query and re-retrieves
  -> If yes but incomplete: Agent identifies gaps and retrieves more
  -> If yes and complete: Return answer
  -> Max iterations: 3-5 (hard limit)
```

#### Planning Pattern (Query Decomposition)

```
Complex query: "Compare the environmental policies of the EU and US in 2024"
  -> Agent decomposes:
     1. "What are the EU environmental policies in 2024?"
     2. "What are the US environmental policies in 2024?"
     3. "What are the key differences between them?"
  -> Each sub-query gets independent retrieval
  -> Results are synthesized into a comprehensive answer
```

LlamaIndex supports this via Sub-Question Query Engine and Query Planning Agents.

#### Tool Use Pattern

```python
# Agent selects tools dynamically based on query analysis
tools = {
    "vector_search": search_vector_db,
    "sql_query": query_structured_db,
    "web_search": tavily_search,
    "calculator": math_tool,
    "api_call": external_api
}

# Agent reasons about which tool to use:
# "This query asks about revenue numbers -> use sql_query"
# "This query asks about recent events -> use web_search"
# "This query asks about internal docs -> use vector_search"
```

#### LlamaIndex Agent Workflows

LlamaIndex (2025-2026) introduced **Agentic Document Workflows (ADW)**:

- **Query-planning agents:** Intelligently route questions to multiple RAG pipelines
- **Sub-question decomposition:** Break complex queries into sub-queries, validate each, merge results
- **Iterative planning:** Create plan of sub-queries, validate before execution, modify plan if results are insufficient (up to 3 iterations)
- **Tool routing:** Agent selects appropriate retrieval tool based on query analysis
- **Recursive retrieval:** Navigate hierarchical document structures

#### Cost Analysis: LLM Calls Per Query

| RAG Pattern | LLM Calls per Query | Retriever Calls | Typical Total Cost |
|-------------|---------------------|-----------------|-------------------|
| Simple RAG | 1 | 1 | $0.01-0.05 |
| RAG + Reranking | 1 | 1 | $0.01-0.07 |
| CRAG | 2-4 | 1-2 | $0.05-0.20 |
| Self-RAG | 3-8 (per segment) | 1-5 | $0.10-0.50 |
| FLARE | 3-10 | 2-5 | $0.10-0.50 |
| Single Agent RAG | 3-6 | 1-3 | $0.10-0.30 |
| Multi-Agent RAG | 5-15 | 2-5 | $0.30-1.00 |

#### Production Deployment Considerations

| Concern | Recommendation |
|---------|---------------|
| **Timeout** | Set API call timeouts (10-30s per LLM call); total pipeline timeout of 60-120s |
| **Max iterations** | Hard limit of 3-5 cycles to prevent infinite loops |
| **Fallback** | Return uncertainty statement rather than hallucination when retrieval exhausted |
| **Cost control** | Set per-query budget; terminate early if exceeded |
| **Observability** | Log every node execution, token counts, retrieval scores for debugging |
| **Caching** | Cache retrieval results and LLM calls for identical/similar queries |
| **Async execution** | Run independent retrieval calls in parallel |
| **State persistence** | Use LangGraph checkpointing for long-running workflows |
| **Error handling** | Graceful degradation: if web search fails, fall back to internal knowledge |

---

### 3.5 Prompt Engineering for RAG

#### System Prompt Best Practices

**Structure:**
```
You are a knowledgeable assistant that answers questions based on provided context.

RULES:
1. Base your answer ONLY on the provided context documents.
2. If the context doesn't contain the answer, say "I cannot find this information in the provided documents."
3. Never fabricate information or use knowledge outside the provided context.
4. Cite your sources by referencing [Source X] where applicable.
5. If the context is unclear or incomplete, state what is missing.

FORMAT:
- Provide concise, direct answers
- Use bullet points for multi-part answers
- Include relevant quotes from source documents
```

**Key principles:**
- Define model role and boundaries explicitly
- Establish grounding rules ("only use provided context")
- Set formatting standards
- Create refusal patterns for missing information
- Separate system prompt (stable rules) from user prompt (dynamic query)

#### Source Attribution Prompting Techniques

**Inline citation:**
```
Answer the question using the context below. For each claim, cite the source
in [brackets] using the format [Source: document_name, page X].
```

**End-of-response citation:**
```
After your answer, include a "Sources" section listing all documents used.
```

**Verbatim quote requirement:**
```
Support each key claim with a direct quote from the source material,
formatted as: "exact quote" (Source: document_name)
```

**Warning:** Research shows up to 57% of LLM citations can be "post-rationalized" -- the model generates the answer first, then finds citations to justify it. Citation faithfulness differs from citation correctness.

#### "Answer From Context Only" Enforcement

**Multi-layer approach:**
1. **System prompt:** "Only use information from the provided context"
2. **Explicit guardrails:** "If the answer cannot be found, do not guess"
3. **Negative instructions:** "Do NOT use your training knowledge to answer"
4. **Transparency request:** "Highlight any assumptions or missing information"
5. **Output constraint:** "Begin your answer with 'Based on the provided documents...'"

**Verification prompt (post-generation):**
```
Review your answer. Is every claim supported by the provided context?
If any claim is not supported, remove it and note what information is missing.
```

#### Chain-of-Thought in RAG

Three variations:

1. **Visible reasoning:** Model shows step-by-step thinking before final answer
   ```
   First, analyze the context to identify relevant information.
   Then, synthesize the information to answer the question.
   Show your reasoning, then provide the final answer.
   ```

2. **Hidden CoT:** Model reasons silently, only final answer displayed
   ```
   Think step-by-step about the context, but only show the final answer.
   ```

3. **Multi-stage workflows:** Sequential prompts for draft, critique, and synthesis
   - Stage 1: Draft answer from context
   - Stage 2: Critique draft for accuracy and completeness
   - Stage 3: Revise and finalize

#### Few-Shot Examples in RAG Prompts

**Placement options:**
- **System prompt level:** For long-term patterns (consistent formatting)
- **User prompt level:** For task-specific or one-time guidance

**Example template:**
```
Example:
Context: "The Eiffel Tower was completed in 1889 and stands 330 meters tall."
Question: "When was the Eiffel Tower built?"
Answer: "The Eiffel Tower was completed in 1889. [Source: context_doc_1]"
```

**Caution:** Longer examples increase token usage and may push retrieval chunks out of the context window. Typically 1-3 examples are sufficient.

#### How Temperature Affects Faithfulness

| Temperature | Faithfulness | Creativity | Recommended Use |
|------------|-------------|------------|-----------------|
| 0.0 | Highest | Lowest | Factoid QA, grounded generation |
| 0.1-0.3 | Very high | Low | Most RAG applications |
| 0.5-0.7 | Moderate | Moderate | Creative summarization |
| 0.8-1.0 | Lower | High | NOT recommended for RAG |

**Research findings:**
- Lower temperature = more deterministic = highest probable next token always picked
- Higher temperature can nudge model toward unlikely, sometimes fabricated tokens
- Google recommends temperature 0.0 for grounded generation
- Best practice: Use 0.0-0.2 for RAG, never exceed 0.5

**Advanced approach (2025):** Best-of-N reranking -- generate multiple candidate responses at slightly higher temperature, then use a factuality metric to select the most faithful one. This outperforms single low-temperature generation in some studies.

---

## Summary: Decision Framework

### When to Use What

| Technique | Best For | Avoid When |
|-----------|----------|------------|
| Cross-encoder reranking | High-accuracy production RAG | Real-time, >1000 QPS without GPU |
| Cohere Rerank | Multilingual, no-GPU, rapid prototyping | High volume (cost) |
| ColBERT | Balanced speed/quality at scale | Small collections (<10K docs) |
| FlashRank | CPU-only, serverless, budget | Maximum accuracy needed |
| LLM reranking | Complex reasoning, zero-shot domains | High throughput, budget |
| LLMLingua | Token/cost savings, long contexts | Short contexts, structured data |
| EXIT compression | Multi-document QA, long contexts | When every sentence matters |
| Self-RAG | Maximum factuality, adaptive retrieval | No fine-tuning budget |
| CRAG | Unreliable retriever, web fallback needed | Simple factoid QA |
| FLARE | Long-form generation, multi-topic | Real-time chat, no logprobs |
| Agentic RAG | Complex multi-step queries | Simple queries (overkill) |

---

## Sources

- [Best Reranker Models for RAG 2026 - BSWEN](https://docs.bswen.com/blog/2026-02-25-best-reranker-models/)
- [Cohere Rerank 3.5 - Oracle Benchmarks](https://docs.oracle.com/en-us/iaas/Content/generative-ai/benchmark-cohere-rerank-3-5.htm)
- [Cohere Pricing](https://cohere.com/pricing)
- [Cohere Rerank Overview](https://docs.cohere.com/docs/rerank-overview)
- [ColBERTv2 Paper](https://arxiv.org/abs/2112.01488)
- [PLAID Engine Paper](https://arxiv.org/pdf/2205.09707)
- [ColBERT Pooling Research - Answer.AI](https://www.answer.ai/posts/colbert-pooling.html)
- [Jina ColBERT v2](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)
- [ColBERT in Practice - Sease](https://sease.io/2025/11/colbert-in-practice-bridging-research-and-industry.html)
- [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- [RankLLM - Castorini](https://github.com/castorini/rank_llm)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [Self-RAG GitHub](https://github.com/AkariAsai/self-rag)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [CRAG LangGraph Tutorial - DataCamp](https://www.datacamp.com/tutorial/corrective-rag-crag)
- [FLARE Paper](https://arxiv.org/abs/2305.06983)
- [FLARE Active RAG - LearnPrompting](https://learnprompting.org/docs/retrieval_augmented_generation/flare-active-rag)
- [LLMLingua - Microsoft](https://github.com/microsoft/LLMLingua)
- [LLMLingua Blog - Microsoft Research](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)
- [EXIT Paper](https://arxiv.org/abs/2412.12559)
- [Lost-in-the-Middle Solutions - GetMaxim](https://www.getmaxim.ai/articles/solving-the-lost-in-the-middle-problem-advanced-rag-techniques-for-long-context-llms/)
- [NVIDIA RULER Benchmark](https://github.com/NVIDIA/RULER)
- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [Building Agentic RAG with LangGraph 2026](https://rahulkolekar.com/building-agentic-rag-systems-with-langgraph/)
- [LangGraph Agentic RAG - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [LlamaIndex Agent Workflows](https://www.llamaindex.ai/workflows)
- [Prompt Engineering for RAG - StackAI](https://www.stackai.com/blog/prompt-engineering-for-rag-pipelines-the-complete-guide-to-prompt-engineering-for-retrieval-augmented-generation)
- [Cross-Encoder Fine-Tuning Guide](https://ranjankumar.in/hands-on-tutorial-fine-tune-a-cross-encoder-for-semantic-similarity)
- [Reranking in RAG - Medium](https://medium.com/@vaibhav-p-dixit/reranking-in-rag-cross-encoders-cohere-rerank-flashrank-c7d40c685f6a)
- [BEIR Benchmark Leaderboard - Ailog](https://app.ailog.fr/en/blog/news/beir-benchmark-update)
- [Databricks Long Context RAG Performance](https://www.databricks.com/blog/long-context-rag-performance-llms)
