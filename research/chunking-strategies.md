# RAG Chunking Strategies: Exhaustive Research Document

> **Research Date**: 2026-04-04
> **Sources**: NVIDIA Technical Blog, Anthropic Research, Jina AI, Chroma Research, Vectara/NAACL 2025, FloTorch 2026, Superlinked VectorHub, AI21 Labs, LangChain Docs, LlamaIndex Docs, Weaviate, various academic papers

---

## Table of Contents

1. [Fixed-Size Chunking](#1-fixed-size-chunking)
2. [Recursive Character Splitting](#2-recursive-character-splitting)
3. [Semantic Chunking](#3-semantic-chunking)
4. [Late Chunking (Jina AI)](#4-late-chunking-jina-ai)
5. [Contextual Chunking (Anthropic)](#5-contextual-chunking-anthropic)
6. [Page-Level Chunking](#6-page-level-chunking)
7. [Agentic Chunking](#7-agentic-chunking)
8. [Document-Specific Chunking](#8-document-specific-chunking)
9. [Proposition-Based Chunking](#9-proposition-based-chunking)
10. [Cross-Cutting Research](#10-cross-cutting-research)

---

## 1. Fixed-Size Chunking

### How It Works (Algorithm Level)

Fixed-size chunking splits documents into chunks of a predetermined size, measured either in characters or tokens. The algorithm is straightforward:

1. Tokenize (or count characters in) the entire document
2. Split at every N tokens/characters
3. Optionally add overlap of M tokens/characters from the previous chunk to the start of the next

**Token-based splitting** uses the model's tokenizer (e.g., `tiktoken` for OpenAI models, `sentencepiece` for others) to count actual tokens. This is more accurate because it matches what the embedding model processes.

**Character-based splitting** counts raw characters (typically ~4 characters per token). It is simpler but can cut words mid-token, causing semantic fragmentation.

### Code-Level Implementation

**LangChain - CharacterTextSplitter:**
```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",        # Split on double newlines first
    chunk_size=1000,         # Characters
    chunk_overlap=200,       # Character overlap
    length_function=len,     # Character count
)
chunks = splitter.split_text(document)
```

**LangChain - TokenTextSplitter:**
```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,          # Tokens
    chunk_overlap=50,        # Token overlap
    encoding_name="cl100k_base"  # tiktoken encoding
)
chunks = splitter.split_text(document)
```

**LlamaIndex - SentenceSplitter (token-based):**
```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
```

### Benchmark Results

| Benchmark | Configuration | Score |
|-----------|--------------|-------|
| FloTorch 2026 (50 academic papers) | Fixed 512 tokens | **67% accuracy** |
| NVIDIA 2024 (FinanceBench) | 1,024 tokens + 15% overlap | **57.9% accuracy** |
| NVIDIA 2024 (Earnings) | 512 tokens + 15% overlap | **68.1% accuracy** |
| NVIDIA 2024 (RAGBattlePacket) | 1,024 tokens + 15% overlap | **80.4% accuracy** |
| Chroma Research | TokenTextSplitter 200 tokens | **87.0% recall** |

### Optimal Configuration Parameters

| Parameter | Recommended Range | Default Starting Point |
|-----------|------------------|----------------------|
| **Chunk size (tokens)** | 128-2048 | 512 tokens |
| **Chunk size (characters)** | 500-8000 | 1000 characters |
| **Overlap (tokens)** | 10-20% of chunk size | 50-100 tokens |
| **Overlap (characters)** | 10-20% of chunk size | 100-200 characters |

**By query type:**
- Factoid queries (names, dates, facts): **256-512 tokens**
- Analytical queries (reasoning, comparison): **512-1024 tokens**
- Multi-hop queries: **1024+ tokens** or page-level

### When to Use

**USE when:**
- Processing homogeneous document collections (encyclopedias, dictionaries)
- Simplicity and speed are priorities over semantic accuracy
- Prototyping or establishing baselines
- Documents lack clear structural markers
- Compute budget is constrained (zero model calls needed)

**DO NOT USE when:**
- Documents have rich structural hierarchy (headers, sections)
- Semantic coherence within chunks is critical
- Documents contain tables, code blocks, or mixed-format content
- Cross-referencing within document is frequent

### Cost Implications

- **Compute**: Near-zero (string splitting only)
- **API calls**: None for chunking itself
- **Storage**: Linear with overlap percentage (20% overlap = ~20% more chunks)

### Edge Cases and Failure Modes

- Character-based splitting can cut words mid-character or mid-token
- Fixed boundaries can split a sentence or paragraph across two chunks, losing context
- Tables and structured data get arbitrarily cut
- Very short documents may produce a single chunk that wastes retrieval slots
- High overlap (>30%) degrades precision without significant recall gains (Chroma research)

---

## 2. Recursive Character Splitting

### How It Works (Algorithm Level)

Recursive character splitting attempts to split text using increasingly granular separators. The algorithm:

1. Start with the highest-priority separator (e.g., `\n\n` for paragraph breaks)
2. Split the text on that separator
3. For each resulting piece, check if it exceeds `chunk_size`
4. If a piece is too large, recurse with the next separator in the hierarchy
5. Continue until all pieces are within `chunk_size` or all separators are exhausted
6. Apply `chunk_overlap` by prepending tokens from the end of the previous chunk

**Default separator hierarchy**: `["\n\n", "\n", " ", ""]`
- `\n\n` = paragraph boundaries (highest priority)
- `\n` = line breaks / sentence boundaries
- `" "` = word boundaries
- `""` = character-level split (last resort)

### Code-Level Implementation

**LangChain - RecursiveCharacterTextSplitter:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,            # or tiktoken-based
    is_separator_regex=False,
    keep_separator=True,
    strip_whitespace=True,
)
chunks = splitter.split_text(document)
```

**Language-specific splitting:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)
# Separators for Python: ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
```

**Supported languages**: CPP, GO, JAVA, KOTLIN, JS, TS, PHP, PROTO, PYTHON, RST, RUBY, RUST, SCALA, SWIFT, MARKDOWN, LATEX, HTML, SOL, CSHARP, COBOL, C, LUA, PERL, HASKELL, ELIXIR, POWERSHELL

### Benchmark Results

| Benchmark | Configuration | Score |
|-----------|--------------|-------|
| **FloTorch 2026** (50 academic papers, 905K tokens) | 512 tokens, 50-100 overlap | **69% accuracy** |
| Chroma Research | 200 tokens, no overlap | **88.1% recall** |
| Chroma Research (all-MiniLM) | 200 tokens | **85.4% recall** |
| Vectara/NAACL 2025 (MSMARCO) | Fixed-size baseline | **93.58% F1@5** |

**FloTorch 2026 finding**: Recursive splitting at 512 tokens scored 69% accuracy, outperforming more expensive semantic chunking (54%) on academic papers. It is the **benchmark-validated default** for most RAG applications.

### Optimal Configuration Parameters

| Parameter | Range | Recommended Default |
|-----------|-------|-------------------|
| `chunk_size` | 256-1024 tokens | **512 tokens** |
| `chunk_overlap` | 0-20% of chunk_size | **50-100 tokens (10-20%)** |
| `separators` (general text) | `["\n\n", "\n", ". ", " ", ""]` | Default hierarchy |
| `separators` (markdown) | `["\n## ", "\n### ", "\n\n", "\n", " "]` | Header-aware |

### When to Use

**USE when:**
- **Default choice for most RAG applications** (benchmark-validated)
- Documents have paragraph/section structure
- Academic papers, blog posts, articles, general prose
- Need a balance between cost (zero model calls) and quality
- Processing Markdown or HTML with clear structure

**DO NOT USE when:**
- Documents are highly cross-referential (consider late chunking)
- Content requires semantic boundary detection (use semantic chunking)
- Documents are paginated PDFs with visual layout (use page-level)

### Cost Implications

- **Compute**: Negligible (string operations only)
- **API calls**: Zero
- **Processing speed**: Milliseconds per document
- **Storage**: ~10-20% overhead from overlap

### Edge Cases and Failure Modes

- Long paragraphs without internal separators force character-level splitting
- Code blocks embedded in prose may get split inappropriately (use language-specific mode)
- Tables without clear newline structure get split arbitrarily
- Separator hierarchy may not match all languages (e.g., CJK text without spaces)

---

## 3. Semantic Chunking

### How It Works (Algorithm Level)

Semantic chunking uses embedding models to detect natural topic boundaries. The core algorithm:

1. Split document into sentences
2. Generate embeddings for each sentence (or groups of 3 sentences using a sliding window)
3. Compute cosine similarity between consecutive sentence embeddings
4. Calculate cosine distance: `distance = 1 - cosine_similarity`
5. Apply a threshold method to identify breakpoints where distance exceeds the threshold
6. Split at those breakpoints

**Threshold methods:**

- **Percentile**: Split where distance exceeds the Xth percentile of all pairwise distances. Default: 95th percentile. Range: 80-95th. Higher = fewer, larger chunks.

- **Standard Deviation**: Split where distance exceeds X standard deviations above the mean. Default: 3.0. Range: 1.0-3.0. Higher = fewer splits.

- **Interquartile**: Uses IQR (Q3-Q1) to identify outlier distances. Split where distance > Q3 + X*IQR. Default: 1.5. Range: 0.5-1.5.

- **Gradient**: Computes the gradient (rate of change) of the distance array, then applies anomaly detection. Useful for highly correlated domain-specific data (legal, medical). Makes the distribution wider and easier to identify boundaries.

### Variants

**Embedding-Similarity Chunking** (KamradtSemanticChunker):
- Cosine distances with percentile thresholds (80th-95th)
- Adaptive to each document's internal structure

**Hierarchical-Clustering Chunking** (ClusterSemanticChunker):
- Uses Within-Cluster Sum of Squares (WCSS) optimization
- Dynamic programming to maximize intra-chunk similarity
- Elbow point method for cluster count

**LLM-Based Semantic Chunking** (LLMSemanticChunker):
- LLM directly identifies split points in text
- Text broken into 50-token segments with XML-style tags
- Model returns split indices (e.g., "split_after: 3, 5")

### Code-Level Implementation

**LangChain - SemanticChunker:**
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",   # or "standard_deviation", "interquartile", "gradient"
    breakpoint_threshold_amount=95,            # percentile value
    buffer_size=1,                             # sentences to group
    add_start_index=False,
    sentence_split_regex=r"(?<=[.?!])\s+",
    min_chunk_size=None,                       # CRITICAL: set this to avoid tiny chunks
)
chunks = text_splitter.split_text(document)
```

**LlamaIndex - SemanticSplitterNodeParser:**
```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding(),
)
```

**Chroma - ClusterSemanticChunker:**
```python
# Implementation from Chroma research
# 1. Split into 50-token pieces via RecursiveCharacterTextSplitter
# 2. Embed each piece individually
# 3. Dynamic programming: maximize cosine similarity sums within chunks
# 4. Enforce max_chunk_size constraint
```

### Benchmark Results

| Method | Benchmark | Metric | Score |
|--------|-----------|--------|-------|
| LLMSemanticChunker | Chroma Research | Recall | **91.9%** |
| ClusterSemanticChunker (400t) | Chroma Research | Recall | **91.3%** |
| ClusterSemanticChunker (200t) | Chroma Research | Precision | **8.0%** (highest) |
| ClusterSemanticChunker (200t) | Chroma Research | IoU | **8.0%** (highest) |
| KamradtModified (300t) | Chroma Research | Recall | **87.1%** |
| Semantic chunking | FloTorch 2026 | Accuracy | **54%** (vs 69% recursive) |
| Breakpoint-based | Vectara/NAACL | F1@5 (Miracl) | **81.89%** |
| Embedding-similarity | Superlinked | Latency | **5.24s** |
| LLM-based | Superlinked | Latency | **6.88s** |

**CRITICAL FINDING**: Semantic chunking achieves high retrieval recall but **fails end-to-end** because it often produces fragments averaging only 43 tokens -- too small for LLMs to generate correct answers. The FloTorch 2026 study showed 54% accuracy vs 69% for recursive splitting. This is the "high retrieval recall, wrong answer" pipeline mismatch.

### Optimal Configuration Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| `breakpoint_threshold_type` | percentile, standard_deviation, interquartile, gradient | **percentile** |
| `breakpoint_threshold_amount` (percentile) | 80-95 | **95** |
| `breakpoint_threshold_amount` (std_dev) | 1.0-3.0 | **3.0** |
| `breakpoint_threshold_amount` (IQR) | 0.5-1.5 | **1.5** |
| `buffer_size` | 1-3 | **1** |
| `min_chunk_size` | 100-300 tokens | **Set this -- essential** |
| Embedding model | Any sentence-level model | BAAI/bge-small-en-v1.5 or text-embedding-3-small |

### When to Use

**USE when:**
- Documents have high topic diversity (stitched or multi-topic documents)
- Retrieval recall is the primary metric and chunk size floor is enforced
- Domain-specific text with clear topical boundaries (legal sections, medical notes)
- Processing long-form narrative content

**DO NOT USE when:**
- Documents are homogeneous or single-topic (adds cost, no benefit)
- End-to-end answer accuracy matters more than recall (use recursive instead)
- Compute budget is constrained (requires embedding every sentence)
- Processing speed matters (5-7 seconds per document vs milliseconds)

### Cost Implications

- **Compute**: Requires embedding every sentence (~$0.01-0.10 per document with API embeddings)
- **API calls**: One embedding call per sentence or sentence group
- **Processing time**: 5-7 seconds per document (vs milliseconds for recursive)
- **Storage**: Variable chunk sizes may require dynamic allocation
- **LLM-based variant**: Tens of minutes per document (Chroma noted), significantly more expensive

### Edge Cases and Failure Modes

- **Tiny chunks**: Without `min_chunk_size`, produces fragments of 20-50 tokens that break downstream generation
- **Homogeneous text**: All distances are similar; threshold methods produce arbitrary boundaries
- **Embedding model mismatch**: Using a different embedding model for chunking vs retrieval can cause misalignment
- **Short documents**: Insufficient sentences for meaningful distance distribution
- **Computational overhead**: Not justified for most use cases per Vectara/NAACL 2025 study

---

## 4. Late Chunking (Jina AI)

### How It Works (Algorithm Level)

Late chunking reverses the traditional order of operations. Instead of "chunk then embed," it "embeds then chunks":

1. **Full document encoding**: Pass the entire document through a long-context transformer model
2. **Token-level representations**: The model produces contextual token embeddings where each token's representation is conditioned on ALL other tokens in the document
3. **Boundary determination**: Apply boundary cues (regex splitting, sentence boundaries) to identify chunk spans
4. **Mean pooling per chunk**: For each chunk span, apply mean pooling over the corresponding token-level vectors
5. **Result**: Each chunk embedding carries contextual information from the entire document

**Key insight**: In traditional chunking, a pronoun like "it" in chunk 5 loses its referent from chunk 2. In late chunking, the token embedding for "it" already encodes the referent because the transformer processed the full document.

### Code-Level Implementation

**Using Jina AI API:**
```python
import requests

url = "https://api.jina.ai/v1/embeddings"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "model": "jina-embeddings-v3",
    "input": ["full document text here"],
    "late_chunking": True          # Enable late chunking
}
response = requests.post(url, headers=headers, json=data)
```

**Manual implementation (research code):**
```python
# From jina-ai/late-chunking GitHub repository
# 1. Tokenize full document
tokens = tokenizer(document, return_tensors="pt", max_length=8192)

# 2. Get token-level embeddings from transformer
with torch.no_grad():
    outputs = model(**tokens)
    token_embeddings = outputs.last_hidden_state  # [1, seq_len, hidden_dim]

# 3. Define chunk boundaries (e.g., by sentence or regex)
chunk_spans = get_chunk_boundaries(document, max_tokens=256)

# 4. Mean pool per chunk span
chunk_embeddings = []
for start, end in chunk_spans:
    chunk_emb = token_embeddings[0, start:end, :].mean(dim=0)
    chunk_embeddings.append(chunk_emb)
```

### Supported Models

| Model | Max Tokens | Context |
|-------|-----------|---------|
| jina-embeddings-v2-base-en | 8,192 tokens | ~10 pages |
| jina-embeddings-v2-small-en | 8,192 tokens | ~10 pages (smaller) |
| **jina-embeddings-v3** | 8,192 tokens | Production API with late chunking |

**Requirement**: Late chunking **requires** long-context embedding models. Standard BERT-based models (512 tokens) are insufficient.

### Benchmark Results (BEIR Datasets, nDCG@10)

| Dataset | Avg Doc Length | Naive Chunking | Late Chunking | Improvement |
|---------|---------------|---------------|--------------|-------------|
| SciFact | 1,498 chars | 64.20% | 66.10% | **+1.9%** |
| TRECCOVID | 1,117 chars | 63.36% | 64.70% | **+1.3%** |
| FiQA2018 | 767 chars | 33.25% | 33.84% | **+0.6%** |
| NFCorpus | 1,590 chars | 23.46% | 29.98% | **+6.5%** |

**Key finding**: Effectiveness increases with document length. NFCorpus (longest average documents) showed the largest improvement.

### Optimal Configuration Parameters

| Parameter | Range | Recommended |
|-----------|-------|------------|
| Chunk span size | 128-512 tokens | **256 tokens** |
| Boundary method | Regex, sentence, paragraph | **Sentence boundaries** |
| Model | jina-embeddings-v2/v3 | **jina-embeddings-v3** |
| Max document length | Up to 8,192 tokens | Model-dependent |

### When to Use

**USE when:**
- Documents have heavy cross-referencing (pronouns referencing earlier content)
- Entity disambiguation is important (same name in different contexts)
- Long technical documents with running context
- Cross-referenced documentation sets

**DO NOT USE when:**
- Documents are short (<1000 tokens) -- minimal benefit
- You need model-agnostic embeddings (locked into Jina ecosystem)
- Documents exceed 8,192 tokens (requires segmentation first)
- Real-time processing required at scale (full document encoding is heavier)

### Cost Implications

- **Compute**: Full document encoding through transformer (heavier than per-chunk encoding)
- **API calls**: One API call per document (vs one per chunk in traditional approach -- actually cheaper per-call)
- **Storage**: Same as traditional chunking (same number of vectors)
- **Vendor lock-in**: Currently requires Jina models (no open alternative for production late chunking)

### Edge Cases and Failure Modes

- Documents exceeding max context window (8,192 tokens) must be segmented first, partially defeating the purpose
- Very short documents show minimal improvement over naive chunking
- Boundary determination still uses heuristics (regex/sentence) -- not semantically optimal
- Not compatible with arbitrary embedding models

### Comparison: Late Chunking vs Contextual Retrieval (Anthropic)

| Aspect | Late Chunking | Contextual Retrieval |
|--------|--------------|---------------------|
| Approach | Embed full doc, then chunk embeddings | Add LLM-generated context to each chunk |
| Model dependency | Requires long-context embedding model | Works with any embedding model |
| Cost per document | One full-doc embedding call | One LLM call per chunk |
| Scalability | More efficient (fewer API calls) | More expensive at scale |
| Context quality | Implicit (from transformer attention) | Explicit (LLM-generated description) |

---

## 5. Contextual Chunking (Anthropic)

### How It Works (Algorithm Level)

Anthropic's Contextual Retrieval prepends a short, LLM-generated context description to each chunk before embedding. The process:

1. **Chunk the document** using any standard method (e.g., recursive, fixed-size)
2. **For each chunk**, send the full document + the chunk to an LLM
3. The LLM generates a 50-100 token context description situating the chunk
4. **Prepend** the context to the chunk text
5. **Embed** the contextualized chunk (for vector search)
6. **Index** the contextualized chunk (for BM25 search)

### Exact Prompt Template

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

**Example output**: "This chunk is from an SEC filing (10-K) by Acme Corp for fiscal year 2023. It describes the revenue recognition policy under ASC 606 for the software licensing segment."

### Code-Level Implementation

```python
import anthropic

client = anthropic.Anthropic()

def generate_context(document: str, chunk: str) -> str:
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk. Answer
only with the succinct context and nothing else."""
        }]
    )
    return response.content[0].text

def contextualize_chunks(document: str, chunks: list[str]) -> list[str]:
    contextualized = []
    for chunk in chunks:
        context = generate_context(document, chunk)
        contextualized.append(f"{context}\n\n{chunk}")
    return contextualized
```

**With prompt caching (90% cost reduction):**
```python
# The document portion is cached between calls
# Only the chunk portion changes per call
# Use Anthropic's prompt caching feature to cache the document prefix
```

### Benchmark Results

**Aggregate retrieval failure reduction (top-20 chunks):**

| Configuration | Failure Rate | Reduction |
|--------------|-------------|-----------|
| Baseline (no context) | 5.7% | -- |
| Contextual Embeddings only | 3.7% | **35% reduction** |
| Contextual Embeddings + BM25 | 2.9% | **49% reduction** |
| Contextual Embeddings + BM25 + Reranking | 1.9% | **67% reduction** |

**Domains tested**: Codebases, fiction, ArXiv papers, science papers

**Retrieval configuration:**
- Top-k retrieval: 150 chunks initially
- Reranking down to top 20 chunks
- Embedding models tested: Gemini Text 004, Voyage (best performers)

### Cost Analysis at Scale

**Assumptions**: 800-token chunks, 8K-token documents, 50-token context instructions, 100 tokens of context per chunk

| Metric | Cost |
|--------|------|
| **Per million document tokens** | **$1.02** (one-time, with prompt caching) |
| Without prompt caching | ~$10.20 per million tokens |
| Per document (8K tokens) | ~$0.008 |
| Per 1M documents | ~$8,160 |
| **With prompt caching savings** | **Up to 90% reduction** |

**Cost breakdown per chunk:**
- Input: 8K (document) + 50 (instruction) + 800 (chunk) = ~8,850 tokens
- Output: ~100 tokens
- With caching: only 800 (chunk) + 100 (output) are non-cached

### Optimal Configuration Parameters

| Parameter | Recommended |
|-----------|------------|
| Base chunking method | Recursive, 512-800 tokens |
| Context model | Claude 3 Haiku (cost-effective) |
| Context length | 50-100 tokens output |
| Retrieval pool | Top 150 chunks |
| Reranking to | Top 20 chunks |
| Embedding model | Voyage or Gemini Text 004 |
| Search method | **Contextual Embeddings + Contextual BM25 + Reranking** |

### When to Use

**USE when:**
- Maximum retrieval accuracy is required and budget allows
- Documents lose important context when chunked (which is most documents)
- Hybrid search (vector + BM25) is already in the pipeline
- One-time indexing cost is acceptable (not for real-time ingestion)
- Domains: legal, financial, scientific, medical documents

**DO NOT USE when:**
- Real-time document ingestion required (LLM call per chunk is slow)
- Cost-constrained environments with large document volumes
- Documents are short enough to be single chunks
- Simple FAQ-style content where chunks are self-contained

### Edge Cases and Failure Modes

- LLM may generate inaccurate or hallucinated context descriptions
- Very long documents exceeding LLM context window require windowing strategies
- Context generation is not deterministic -- same chunk may get different contexts on re-indexing
- Batch processing needed for efficiency (async processing recommended)
- If the LLM model is updated, existing contexts may become stylistically inconsistent

---

## 6. Page-Level Chunking

### How It Works (Algorithm Level)

Page-level chunking uses document pagination as the sole chunking boundary:

1. Parse the document to identify page breaks
2. Each page becomes a single chunk
3. No overlap between chunks (pages are discrete units)
4. Page metadata (page number, section title) can be attached

### Code-Level Implementation

**Using NVIDIA NeMo Retriever:**
```python
# NVIDIA's approach uses nemoretriever-parse for page extraction
# Each page is extracted with layout awareness
```

**Using PyMuPDF:**
```python
import fitz  # PyMuPDF

def page_level_chunks(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        chunks.append({
            "text": text,
            "metadata": {
                "page": page_num + 1,
                "source": pdf_path
            }
        })
    return chunks
```

**Using Unstructured.io:**
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res",          # Uses layout detection
    chunking_strategy="by_page",
)
```

### Benchmark Results (NVIDIA 2024)

**Methodology:**
- **Datasets**: DigitalCorpora767 (767 PDFs, 991 questions), Earnings (512 PDFs, 600+ questions), FinanceBench, KG-RAG, RAGBattlePacket
- **Metric**: NV Answer Accuracy (RAGAS framework, 0-4 scale)
- **Judge models**: Mixtral 8x22B Instruct, Llama 3.1 70B Instruct
- **Embedding**: nvidia/llama-3.2-nv-embedqa-1b-v2
- **Reranker**: nvidia/llama-3.2-nv-rerankqa-1b-v2
- **Top-k**: 10

| Strategy | Average Accuracy | Standard Deviation |
|----------|-----------------|-------------------|
| **Page-level (NeMo)** | **0.648** | **0.107** (lowest) |
| Token 2048 | 0.645 | 0.115 |
| Token 1024 | 0.635 | 0.131 |
| Token 512 | 0.624 | 0.144 |
| Token 256 | 0.613 | 0.148 |
| Token 128 | 0.603 | 0.157 |
| Section-level | 0.618 | 0.142 |

**Page-level had both the highest average accuracy and the most consistent performance (lowest standard deviation).**

### Optimal Configuration

| Parameter | Value |
|-----------|-------|
| Extraction method | Layout-aware parser (NeMo, Unstructured hi_res) |
| Page metadata | Page number, section headers, document title |
| Max page size | ~500-1500 tokens (typical PDF page) |
| Overlap | None (pages are natural boundaries) |

### When to Use

**USE when:**
- **Paginated documents**: Financial reports, annual filings (10-K, 10-Q), legal contracts
- Documents with page-level visual organization (tables, figures, charts)
- Information is self-contained per page by design
- Consistent, low-variance performance is needed across diverse datasets

**DO NOT USE when:**
- Documents are not paginated (plain text, markdown, HTML)
- Content flows freely across page boundaries (novels, long-form prose)
- Pages have highly variable content density (some pages nearly empty)
- Individual pages exceed embedding model token limits

### Cost Implications

- **Compute**: PDF parsing with layout detection (moderate)
- **API calls**: Zero for chunking itself
- **Storage**: Number of chunks = number of pages (predictable)
- **Processing**: Layout-aware parsing (nemoretriever-parse) is heavier than simple text extraction

### Edge Cases and Failure Modes

- Scanned PDFs require OCR before page-level chunking
- Multi-column pages may extract text in wrong order
- Tables spanning multiple pages lose context
- Very dense pages may exceed embedding model token limits
- Near-empty pages (title pages, separator pages) create low-information chunks

---

## 7. Agentic Chunking

### How It Works (Algorithm Level)

Agentic chunking uses an LLM as a decision-maker to dynamically determine chunk boundaries:

1. **Break text into atomic propositions** (individual factual statements)
2. **For each proposition**, the LLM decides:
   - Does this belong to an existing chunk? (Compare with chunk summaries)
   - Should it start a new chunk? (New topic detected)
3. The LLM **generates metadata** for each chunk: title, summary, keywords
4. Chunks are enriched with AI-generated annotations before indexing

**Four LLM responsibilities:**
- Analyze content and understand semantic meaning
- Compare adjacent propositions for relatedness
- Decide if a boundary should exist
- Adapt decisions based on evolving context

### Code-Level Implementation

**Using Phidata/Agno:**
```python
from phi.agent import Agent
from phi.document.chunking.agentic import AgenticChunking
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://example.com/document.pdf"],
    vector_db=PgVector(table_name="agentic_chunks", db_url=db_url),
    chunking_strategy=AgenticChunking(
        model=OpenAIChat(),        # LLM for chunking decisions
        max_chunk_size=5000,       # Upper limit for chunk length
    ),
)
knowledge_base.load(recreate=False)
```

**Using LangChain + custom implementation:**
```python
# Typical agentic chunking pipeline:
# 1. Split into propositions
# 2. For each proposition, ask LLM to classify into existing or new chunk

def agentic_chunk(text: str, llm) -> list[dict]:
    propositions = extract_propositions(text, llm)
    chunks = []
    current_chunk = {"title": "", "summary": "", "propositions": []}

    for prop in propositions:
        # Ask LLM: does this belong to current chunk?
        decision = llm.classify(prop, current_chunk["summary"])
        if decision == "new_chunk":
            chunks.append(current_chunk)
            current_chunk = {"title": "", "summary": "", "propositions": [prop]}
            current_chunk["title"] = llm.generate_title(prop)
        else:
            current_chunk["propositions"].append(prop)
        current_chunk["summary"] = llm.summarize(current_chunk["propositions"])

    chunks.append(current_chunk)
    return chunks
```

### Configuration Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `model` | LLM | OpenAI GPT-4 | Any instruction-following LLM |
| `max_chunk_size` | int | 5000 chars | 1000-10000 |
| Proposition extraction | LLM call | -- | GPT-4 or finetuned model |

### Benchmark Results

No standardized benchmarks exist specifically for agentic chunking. Performance is inferred from its components:
- Proposition extraction quality depends on the LLM used
- Chunk coherence is generally higher than any rule-based method
- Processing time: orders of magnitude slower than alternatives

### When to Use

**USE when:**
- Highest possible chunk quality justifies extreme cost
- High-stakes domains: medical, legal, financial compliance
- Document structure is completely unpredictable
- RAG system serves critical decision-making (clinical decision support)
- One-time indexing of a small, high-value corpus

**DO NOT USE when:**
- Large document volumes (cost and time prohibitive)
- Real-time ingestion required
- Budget is constrained (requires LLM call per proposition)
- Simpler methods achieve acceptable quality (which they usually do)

### Cost Implications

- **Compute**: Multiple LLM calls per document (most expensive method)
- **API calls**: ~1 call per paragraph (proposition extraction) + ~1 call per proposition (classification) + summary updates
- **Estimated cost**: $0.50-5.00 per document (depending on length and model)
- **Processing time**: Minutes to tens of minutes per document
- **Storage**: Enriched metadata increases storage requirements

### Edge Cases and Failure Modes

- LLM inconsistency: Same document may produce different chunks on re-run
- Cost explosion on large documents (hundreds of LLM calls)
- LLM hallucination in metadata/summaries can poison retrieval
- Proposition extraction may miss implicit information
- No standardized evaluation framework

---

## 8. Document-Specific Chunking

### PDF Documents

**Challenge**: PDFs are visual formats. Text extraction is unreliable due to columns, headers, footers, tables, and scanned content.

**Best approach**: Convert PDF to structured format (Markdown) first, then apply recursive chunking.

```python
# Using Unstructured.io
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res",              # Layout detection
    infer_table_structure=True,     # Detect tables
    extract_images_in_pdf=True,     # Extract embedded images
)

# Using Docling (IBM)
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
markdown_text = result.document.export_to_markdown()
```

**Table handling**: When tables span multiple chunks, repeat table headers at the beginning of each continuation chunk to preserve column context.

### Code Files

**LangChain language-specific splitting:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)
# Separators: ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# JavaScript
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=2000,
    chunk_overlap=200,
)
```

**Supported languages**: Python, JavaScript, TypeScript, Java, C++, C#, Go, Rust, Ruby, PHP, Scala, Swift, Kotlin, Lua, Perl, Haskell, Elixir, Cobol, Markdown, LaTeX, HTML, RST, Solidity, PowerShell, Visual Basic 6

**Best practice**: Keep functions, classes, and methods as atomic units. Never split a function body across chunks.

### Markdown Files

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)
chunks = splitter.split_text(markdown_text)
# Each chunk includes its header hierarchy as metadata
```

**Strategy**: Use H2 sections as primary chunks. Group related H3 subsections if they fit within token limits.

### HTML Documents

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_content)
```

**Best practice**: Split by semantic tags (`<article>`, `<section>`, `<p>`, `<div>`). Remove boilerplate (nav, footer, sidebar) before chunking.

### Legal Documents

**Strategy**: Split by clause/section structure.
- Detect headings: "Section 4.2 Liability", "Article III", "Clause 7(a)"
- Each clause becomes a chunk
- Cross-references between clauses should be preserved in metadata
- **Recommended**: Recursive splitting with legal-specific separators + contextual chunking (Anthropic) for clause context

### Tables (in PDFs/Documents)

**Using Docling HybridChunker:**
```python
from docling.chunking import HybridChunker

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=512,
    merge_peers=True,              # Merge small adjacent chunks
    # Table headers are automatically repeated when tables span chunks
)
chunks = list(chunker.chunk(document))
```

**Key feature**: Table headers are repeated at the beginning of each chunk when a table spans multiple chunks, ensuring each chunk maintains column context.

### Recommendations by Document Type

| Document Type | Primary Strategy | Chunk Size | Special Handling |
|--------------|-----------------|-----------|-----------------|
| Academic papers | Recursive | 512 tokens | Section-aware separators |
| Financial PDFs | **Page-level** | Per-page | Layout-aware parsing |
| Code files | Language-specific recursive | 2000 chars | Function-level boundaries |
| Markdown docs | Header-based | 512 tokens/section | Header hierarchy metadata |
| HTML pages | Tag-based + recursive | 512 tokens | Remove boilerplate first |
| Legal contracts | Clause-based + contextual | 512-1024 tokens | Cross-reference metadata |
| FAQ pages | No chunking | Full Q&A pair | Each Q&A is a chunk |
| Chat logs | Message-based | Per conversation turn | Preserve speaker metadata |
| Tables | Hybrid (Docling) | 512 tokens | Repeat headers across chunks |

---

## 9. Proposition-Based Chunking (Dense-X Retrieval)

### How It Works (Algorithm Level)

From the paper "Dense X Retrieval: What Retrieval Granularity Should We Use?" (Chen et al., 2023):

1. **Decompose paragraphs** into atomic propositions using a fine-tuned LLM
2. Each proposition is a **self-contained, atomic, verifiable statement of fact**
3. **Properties of a valid proposition:**
   - **Unique**: Represents a distinct piece of meaning
   - **Atomic**: Cannot be further split into separate propositions
   - **Self-contained**: Includes all necessary context (no dangling pronouns)
4. Index propositions individually for dense retrieval
5. At retrieval time, match query against proposition-level embeddings
6. Return the parent passage/document for LLM context

### Code-Level Implementation

**Using LlamaIndex DenseXRetrieval:**
```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.packs.dense_x_retrieval import DenseXRetrievalPack

# The pack handles:
# 1. Document -> passage splitting
# 2. Passage -> proposition extraction (via LLM)
# 3. Proposition-level indexing
# 4. Retrieval with parent-passage linking

pack = DenseXRetrievalPack(
    documents=documents,
    proposition_llm=OpenAI(model="gpt-4"),
    query_llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding(),
)
response = pack.run("What is the capital of France?")
```

**Manual proposition extraction:**
```python
def extract_propositions(paragraph: str, llm) -> list[str]:
    prompt = """Decompose the following text into clear, atomic propositions.
Each proposition should be:
1. A single, self-contained statement of fact
2. Understandable without external context
3. Cannot be further decomposed

Text: {text}

Return each proposition on a new line."""

    response = llm.generate(prompt.format(text=paragraph))
    return [p.strip() for p in response.split("\n") if p.strip()]
```

**Original research model**: Fine-tuned FlanT5-large, trained on 42K passages atomized into propositions using GPT-4.

### Benchmark Results

**From Dense-X Retrieval paper:**

| Metric | Improvement | Dataset |
|--------|------------|---------|
| Recall@5 (unsupervised, DPR) | **+17-25% relative** | EntityQuestions |
| Recall@20 (unsupervised) | **+10.1%** | General benchmarks |
| Exact Match (QA) | **+4.9 to +7.8** | Multiple datasets |

**Optimal granularity**: ~10 propositions = ~100-200 words = ~5 sentences = ~2 traditional passages

**Key finding**: Propositions have the highest density of relevant information compared to passages or sentences. The LLM generates better answers when propositions are injected as context.

**Particular strength**: Questions targeting **less common entities** show dramatically better performance with proposition-level retrieval.

### Configuration Parameters

| Parameter | Range | Recommended |
|-----------|-------|------------|
| Proposition extraction model | GPT-4, FlanT5-large (finetuned) | GPT-4 for quality, finetuned for cost |
| Propositions per passage | 5-15 | ~10 per 100-200 words |
| Index granularity | Proposition-level | With parent passage linking |
| Retrieval top-k | 5-20 | 10 propositions, map to top-5 passages |

### When to Use

**USE when:**
- Factoid questions are the primary query type
- Entity-centric retrieval (people, places, dates, facts)
- Knowledge base construction from diverse sources
- Questions target rare or uncommon entities
- Maximum retrieval precision is worth the compute cost

**DO NOT USE when:**
- Analytical or reasoning queries (need broader context)
- Budget constraints (LLM call per paragraph)
- Real-time ingestion pipelines
- Documents are already well-structured at the sentence level

### Cost Implications

- **Compute**: One LLM call per paragraph for proposition extraction
- **Storage**: ~5-10x more index entries than passage-level (each passage produces ~10 propositions)
- **Retrieval cost**: More comparisons at query time (larger index)
- **Estimated cost**: $0.10-1.00 per document for extraction (model-dependent)
- **One-time cost**: Can use fine-tuned smaller model (FlanT5-large) to reduce ongoing costs

### Edge Cases and Failure Modes

- Proposition extraction can miss implicit information or nuance
- Self-containment requirement may produce verbose propositions
- High index size (10x) increases retrieval latency
- Some facts are inherently relational and resist atomization
- Quality depends heavily on the extraction LLM

---

## 10. Cross-Cutting Research

### NVIDIA 2024 Benchmark: Full Methodology

**Study**: "Finding the Best Chunking Strategy for Accurate AI Responses" (NVIDIA Technical Blog)

**Setup:**
- 5 diverse datasets: DigitalCorpora767, Earnings, FinanceBench, KG-RAG, RAGBattlePacket
- 7 chunking strategies: Token-based (128, 256, 512, 1024, 2048), page-level, section-level
- All strategies used 15% overlap for token-based approaches
- Standardized pipeline: nvidia/llama-3.2-nv-embedqa-1b-v2 embedding + nvidia/llama-3.2-nv-rerankqa-1b-v2 reranking + top-10 retrieval
- Metric: NV Answer Accuracy (RAGAS framework)
- Judge models: Mixtral 8x22B Instruct, Llama 3.1 70B Instruct

**Results summary:**

| Strategy | Avg Accuracy | Std Dev | Best For |
|----------|-------------|---------|----------|
| Page-level | **0.648** | **0.107** | Overall winner, lowest variance |
| Token 2048 | 0.645 | 0.115 | Large analytical queries |
| Token 1024 | 0.635 | 0.131 | Financial documents |
| Token 512 | 0.624 | 0.144 | General purpose |
| Token 256 | 0.613 | 0.148 | Factoid queries |
| Token 128 | 0.603 | 0.157 | Short factual lookups |
| Section-level | 0.618 | 0.142 | Structured documents |

### Vectara/NAACL 2025: Chunking vs Embedding Model

**Study**: "Is Semantic Chunking Worth the Computational Cost?" (Vectara, NAACL 2025 Findings)

**Setup:**
- 3 chunker types (fixed, breakpoint-semantic, cluster-semantic) with 25+ configurations
- 3 embedding models: stella_en_1.5B (Rank 3), bge-large-en-v1.5 (Rank 36), all-mpnet-base-v2 (Rank 105)
- 10 document retrieval datasets + 5 RAGBench datasets for evidence retrieval + answer generation
- Metric: F1@5

**Key findings:**
1. **Chunking configuration has as much or more influence on retrieval quality as the choice of embedding model**
2. Fixed-size chunking **consistently outperformed** semantic chunking on original (non-stitched) datasets
3. Breakpoint-based semantic chunking excelled on **stitched datasets with high topic diversity** (81.89% vs 69.45% on Miracl)
4. Benefits of semantic chunking were **highly context-dependent** and did not consistently justify computational cost
5. Evidence retrieval differences between methods were **minimal** (47.11% vs 47.08% on ExpertQA)
6. Answer generation (BERTScore): differences too small for definitive conclusions

**Conclusion**: "Most teams tune their embedding model obsessively and ignore how the documents were split. That is backwards."

### Chroma Research: Token-Level Evaluation

**Study**: "Evaluating Chunking Strategies for Retrieval" (Chroma, July 2024)

**Setup:**
- 328,208 tokens across 5 corpora (State of Union, Wikitext, Chatlogs, Finance, PubMed)
- 472 queries (GPT-4 generated)
- Token-level metrics: Recall, Precision, IoU (Jaccard at token level)
- Primary model: text-embedding-3-large, validated on all-MiniLM-L6-v2

**Full results (text-embedding-3-large, top-5 chunks):**

| Strategy | Avg Tokens | Recall | Precision | IoU |
|----------|-----------|--------|-----------|-----|
| LLMChunker | ~240 | **91.9%** | 3.9% | 3.9% |
| ClusterSemantic (400t) | ~400 | 91.3% | 4.2% | 4.2% |
| RecursiveCharacter (200t) | 200 | 88.1% | 7.0% | 7.0% |
| ClusterSemantic (200t) | ~103 | 87.3% | **8.0%** | **8.0%** |
| TokenText (200t) | 200 | 87.0% | 5.2% | 5.1% |
| KamradtModified (300t) | ~397 | 87.1% | 2.1% | 2.1% |

**Key insight**: OpenAI's recommended defaults (800 tokens, 400 overlap) achieved only **1.5% precision and 1.5% IoU** -- described as "particularly poor recall-efficiency tradeoffs."

### Chunk Size vs Query Type

| Query Type | Optimal Chunk Size | Why |
|-----------|-------------------|-----|
| **Factoid** (names, dates, facts) | 256-512 tokens | Answer is localized; smaller chunks = less noise |
| **Analytical** (reasoning, comparison) | 512-1024 tokens | LLM needs broader context to synthesize |
| **Multi-hop** (connecting multiple facts) | 1024+ tokens or page-level | Multiple relevant facts often co-occur |
| **Summarization** | Full document or large chunks | Need comprehensive coverage |

**AI21 Labs research on multi-scale chunking:**
- Index same corpus at multiple sizes (50, 100, 200, 500, 1000, 2000 tokens)
- Aggregate with Reciprocal Rank Fusion (RRF)
- Improvement: **1-37% across benchmarks** (TRECCOVID: +36.7%)
- Oracle experiments show **20-40% headroom** when selecting optimal chunk size per query
- Tradeoff: 2-5x indexing and storage cost

### Chunk Overlap Optimization

**Consensus from research:**

| Overlap | Effect |
|---------|--------|
| 0% | Fastest indexing, lowest storage, may miss boundary context |
| 10-20% (recommended) | Best precision-recall tradeoff |
| >30% | Degrades precision without significant recall gains |

**Specific finding (Chemistry RAG study)**: Non-overlapping recursive chunking (R100-0) was a strong default, offering excellent performance with minimal indexing overhead.

**NVIDIA approach**: 15% overlap consistently across all token-based strategies.

**Rule of thumb**: For 512-token chunks, use 50-100 tokens of overlap.

### Metadata Enrichment Strategies

**Types of metadata to add to chunks:**

| Category | Examples |
|----------|---------|
| **Content-based** | Keywords, summary, topic, named entities, domain tags |
| **Structural** | Section headers, page numbers, TOC position, heading level |
| **Contextual** | Source system, ingestion date, data sensitivity, language |
| **Relational** | Parent document ID, sibling chunk IDs, cross-references |

**MetaRAG framework** (Enterprise Knowledge Retrieval, 2025):
- LLM-generated metadata: topic/category, time period, entities, metrics, document type
- **Metadata proportion**: 10% metadata consistently improves retrieval (acts as "seasoning effect")
- Higher metadata ratios degrade performance
- **Prefix-fusion** embedding approach: Concatenate metadata as prefix before encoding, letting the encoder contextually modulate metadata influence
- Production viable: **sub-30ms P95 latency** with metadata-enriched retrieval

**MDKeyChunker** (2025):
- Single LLM call per document for metadata enrichment
- Rolling key extraction with key-based restructuring
- Designed for high-accuracy RAG at scale

### How to Choose Chunking Strategy by Document Type

**Decision flowchart:**

```
Is the document paginated (PDF)?
├── Yes: Is it a financial/structured report?
│   ├── Yes → PAGE-LEVEL CHUNKING
│   └── No → Extract to markdown, then RECURSIVE (512 tokens)
│
└── No: Is it code?
    ├── Yes → LANGUAGE-SPECIFIC RECURSIVE
    └── No: Is it markdown/HTML?
        ├── Yes → HEADER-BASED SPLITTING
        └── No: Do you need maximum retrieval accuracy?
            ├── Yes: Budget allows LLM calls per chunk?
            │   ├── Yes → CONTEXTUAL CHUNKING (Anthropic)
            │   └── No → RECURSIVE (512 tokens) + metadata enrichment
            └── No → RECURSIVE (512 tokens, 50-100 overlap)
```

**Universal baseline**: Recursive character splitting at 512 tokens with 50-100 tokens of overlap. This is the benchmark-validated default for most RAG applications, requires zero model calls, runs in milliseconds, and outperformed more expensive alternatives in the largest 2026 real-document test (FloTorch).

### Strategy Comparison Matrix

| Strategy | Accuracy | Speed | Cost | Complexity | Best Use Case |
|----------|---------|-------|------|-----------|--------------|
| Fixed-size | Medium | Fastest | Free | Lowest | Baseline, homogeneous docs |
| Recursive | **High** | Fastest | Free | Low | **Universal default** |
| Semantic | Medium (e2e) | Slow | Medium | Medium | Multi-topic documents |
| Late chunking | High | Medium | Low | Medium | Cross-referential docs |
| Contextual (Anthropic) | **Highest** | Slow | High | Medium | High-stakes domains |
| Page-level | **High** | Fast | Free | Low | Paginated PDFs |
| Agentic | High | **Slowest** | **Highest** | Highest | Small high-value corpora |
| Proposition-based | High (factoid) | Slow | High | High | Entity-centric QA |
| Document-specific | Varies | Varies | Low-Medium | Medium | Mixed document types |

---

## Summary of Key Takeaways

1. **Start with recursive splitting at 512 tokens, 50-100 overlap**. This outperformed expensive semantic chunking in the largest 2026 benchmark (69% vs 54% accuracy).

2. **Chunking strategy matters as much as embedding model choice** (Vectara/NAACL 2025). Most teams optimize the wrong thing.

3. **Page-level chunking wins for paginated PDFs** (NVIDIA 2024: 0.648 accuracy, lowest variance).

4. **Semantic chunking's high recall is misleading** -- it produces tiny fragments that break end-to-end accuracy. Always set a `min_chunk_size` floor.

5. **Contextual chunking (Anthropic) achieves the best retrieval quality** but at $1.02/million tokens (with caching). Worth it for high-stakes domains.

6. **Late chunking is the most elegant solution** for cross-chunk context loss, but requires Jina ecosystem lock-in.

7. **Proposition-based chunking excels for factoid/entity queries** with 17-25% recall improvement, but 10x storage increase.

8. **Multi-scale chunking** (indexing at multiple sizes with RRF fusion) shows 1-37% improvement but 2-5x storage cost.

9. **Metadata enrichment at 10% proportion** acts as a "seasoning effect" that consistently improves retrieval with sub-30ms latency impact.

10. **No single strategy wins everywhere**. The optimal approach depends on document type, query type, and budget constraints.
