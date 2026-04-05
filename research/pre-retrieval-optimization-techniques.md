# Pre-Retrieval Optimization Techniques in RAG Systems
## Exhaustive Research Document

**Date:** 2026-04-04
**Scope:** All major pre-retrieval optimization techniques for Retrieval-Augmented Generation systems

---

## Table of Contents

1. [Query Rewriting / Multi-Query Expansion](#1-query-rewriting--multi-query-expansion)
2. [HyDE (Hypothetical Document Embeddings)](#2-hyde-hypothetical-document-embeddings)
3. [Query Routing](#3-query-routing)
4. [Query Decomposition](#4-query-decomposition)
5. [Query Classification / Intent Detection](#5-query-classification--intent-detection)
6. [Query Expansion with Pseudo-Relevance Feedback (PRF)](#6-query-expansion-with-pseudo-relevance-feedback-prf)
7. [Combining Techniques](#7-combining-techniques)
8. [Production Deployment Patterns](#8-production-deployment-patterns)
9. [Evaluation and A/B Testing](#9-evaluation-and-ab-testing)

---

## 1. Query Rewriting / Multi-Query Expansion

### 1.1 How LLM-Based Query Rewriting Works

Query rewriting is a pre-retrieval strategy that bridges the gap between user input and the knowledge required by the retrieval system by rephrasing the original query. The process:

1. **User submits a query** (often conversational, ambiguous, or underspecified)
2. **LLM receives the query** along with a rewriting prompt template
3. **LLM generates one or more reformulated queries** optimized for retrieval
4. **Reformulated queries are embedded** and sent to the vector store
5. **Results are merged and deduplicated** across all query variants
6. **Top-K documents** are returned to the generation stage

The key insight: query quality often determines retrieval performance, especially critical for large knowledge bases (~1,000,000+ documents).

### 1.2 Different Rewriting Strategies

**a) Paraphrasing / Synonym Expansion**
- Replaces terms with synonyms to match different vocabulary in the corpus
- Example: "How to fix a bug" -> "How to debug an error", "Troubleshooting software defects"

**b) Expansion / Enrichment**
- Adds missing keywords, context, and specificity
- Removes conversational language
- Example: "What's that thing with React?" -> "What is the React virtual DOM reconciliation algorithm?"

**c) Specialization**
- Narrows broad queries to specific retrievable topics
- Example: "Tell me about AI" -> "What are the current applications of transformer-based large language models in enterprise NLP?"

**d) Multi-perspective Generation**
- Generates queries from different viewpoints (user vs expert, narrow vs broad)
- Covers vocabulary variation, technical terminology, and different conceptual angles

**The DMQR-RAG Framework** (arXiv:2411.13154) introduced four distinct rewriting strategies:
- **GQR (General Query Rewriting):** Refines a query by omitting noise and maintaining relevant information
- **KWR (Keyword Rewriting):** Extracts keywords that search engines prefer
- **PAR (Pseudo Answer Rewriting):** Constructs a pseudo answer to broaden the query with useful information
- **CCE (Core Concept Extraction):** Concentrates on finding key information where detailed queries are extracted

DMQR-RAG also proposes an **adaptive strategy selection method** that minimizes the number of rewrites while optimizing overall performance. Results: ~8% improvement on HotpotQA; surpasses HyDE on AmbigNQ by +1.30% EM and +3.74% F1.

**Sources:**
- [DMQR-RAG Paper](https://arxiv.org/abs/2411.13154)
- [Query Rewrite in RAG Systems - DEV Community](https://dev.to/yaruyng/query-rewrite-in-rag-systems-why-it-matters-and-how-it-works-3mmd)

### 1.3 LangChain MultiQueryRetriever Implementation

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Basic usage
retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    include_original=True  # Include original query in the list
)

docs = retriever.get_relevant_documents(query=question)
```

**Default Prompt Template** (built into LangChain):
The default instructs the AI to generate 3 different versions of a user question to retrieve relevant documents from a vector database, overcoming limitations of distance-based similarity search, with alternatives separated by newlines.

**Custom Prompt Template:**
```python
from langchain.prompts import PromptTemplate

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a search query optimizer.
Rewrite the user's question to improve retrieval quality.
Rules:
1. Preserve original meaning
2. Remove conversational language
3. Add missing keywords
4. Generate 3 different search queries

Original question: {question}

Provide these alternative questions separated by newlines:"""
)

retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    prompt=CUSTOM_PROMPT
)
```

**Sources:**
- [LangChain MultiQueryRetriever API](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html)
- [LangChain How-To Guide](https://python.langchain.com/v0.2/docs/how_to/MultiQueryRetriever/)
- [Multi-Query Retriever RAG - DEV Community](https://dev.to/sreeni5018/multi-query-retriever-rag-how-to-dramatically-improve-your-ais-document-retrieval-accuracy-5892)

### 1.4 LlamaIndex Query Transformation Implementations

LlamaIndex provides query transformations through its **Sub Question Query Engine**:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="documents",
            description="Useful for answering questions about the documents"
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
)
```

### 1.5 Number of Rewrites vs Performance (Diminishing Returns)

| Number of Variations | Impact |
|---------------------|--------|
| 1 (baseline) | Standard single-query retrieval |
| 3 variations | Minimum for meaningful coverage improvement |
| 4-5 variations | **Sweet spot** for accuracy/cost balance |
| >5 variations | Diminishing returns with increased latency and cost |

**Benchmark data from a 378-document technical corpus:**

| Query Type | Simple RAG Miss Rate | Multi-Query Improvement |
|-----------|---------------------|------------------------|
| Specific | 5-10% | ~0% miss rate |
| Compound | 15-25% | +3-5% improvement |
| Abstract | 30-40% | +5-10% improvement |
| Ambiguous | Very poor | +15-25% improvement |

Multi-Query found **8 unique documents vs. 3 from simple RAG** (166% more coverage) for complex queries.

**Source:** [Multi-Query Retriever RAG - DEV Community](https://dev.to/sreeni5018/multi-query-retriever-rag-how-to-dramatically-improve-your-ais-document-retrieval-accuracy-5892)

### 1.6 Which LLM Models Work Best for Rewriting

| Model | Use Case | Cost | Quality |
|-------|----------|------|---------|
| GPT-4o / GPT-4 | Highest quality rewrites | High ($) | Best |
| GPT-4o-mini | High-volume production | Very Low ($) | Very Good |
| GPT-3.5-Turbo | Budget rewriting | Low ($) | Good |
| Claude 3 Haiku / Claude 4.5 Haiku | Cost-optimized | $0.15-0.60/M tokens | Good |
| Llama 3 8B (via Groq) | Self-hosted / lowest cost | $0.20-0.90/M tokens | Adequate |
| Qwen2-7B / Llama3-8B | On-prem, no API cost | Infrastructure only | Adequate |

**Key finding from DMQR-RAG:** The framework can be effectively applied to much smaller LLMs (Llama3-8B and Qwen2-7B) rather than requiring GPT-4, making it production-viable at lower cost.

**Cost rule of thumb:** GPT-3.5 instead of GPT-4 reduces cost by 10x and improves speed by 3x+ while maintaining good rewrite quality. GPT-3.5 achieves 0.9048 accuracy on RAG tasks versus Mistral-7b at 0.857.

**Sources:**
- [LLM API Cost Comparison](https://inventivehq.com/blog/llm-api-cost-comparison)
- [RAG Cost Optimization Strategies](https://zenvanriel.com/ai-engineer-blog/rag-cost-optimization-strategies/)
- [DMQR-RAG Paper](https://arxiv.org/abs/2411.13154)

### 1.7 Prompt Templates for Query Rewriting

**General-Purpose Rewriting Template:**
```python
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries
to improve retrieval in a RAG system. Given the original query, rewrite it to be more
specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""
```

**Multi-Query Generation Template:**
```python
multi_query_template = """You are a search query optimizer.
Rewrite the user's question to improve retrieval quality.
Rules:
1. Preserve original meaning
2. Remove conversational language
3. Add missing keywords
4. Generate 3 different search queries covering:
   - Different vocabulary (synonyms, technical terms)
   - Different perspectives (user vs. expert viewpoints)
   - Different scope (narrow vs. broad interpretations)

Original question: {question}

Return as JSON with "intent" and "queries" array."""
```

**Source:** [RAG Techniques Repository](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)

### 1.8 Benchmarks: Improvement in Recall/Precision

- **Precision improvements:** 25-50% reported across enterprise deployments
- **Recall improvements:** 15-35%
- **End-to-end user satisfaction:** +20-40%
- **DMQR-RAG on HotpotQA:** ~8% improvement in multi-query vs single-query rewriting
- **Reinforcement learning for query optimization:** 15-25% additional precision improvements beyond static rewriting (early 2025 research)

**Source:** [The Query Rewriting Revolution](https://ragaboutit.com/the-query-rewriting-revolution-how-smart-prompt-engineering-is-eliminating-rag-retrieval-failures/)

### 1.9 Cost Analysis: Additional LLM Calls Per Query

| Technique | Additional LLM Calls | Estimated Added Cost (GPT-4o-mini) | Added Latency |
|-----------|----------------------|-------------------------------------|---------------|
| Single rewrite | 1 | ~$0.0001-0.0003 | 200-500ms |
| Multi-query (3 variants) | 1 (generates all at once) | ~$0.0002-0.0005 | 300-700ms |
| Multi-query (5 variants) | 1 | ~$0.0003-0.0008 | 400-900ms |
| Adaptive DMQR-RAG | 1-4 (adaptive) | ~$0.0005-0.002 | 500-2000ms |

**Pipeline stages:** User Query -> Query Rewrite -> Intent Analysis -> Multi Retrieval -> Hybrid Merge -> Top-K -> Score Threshold -> Rerank -> LLM

---

## 2. HyDE (Hypothetical Document Embeddings)

### 2.1 Original Paper Methodology (arXiv:2212.10496)

**Paper:** "Precise Zero-Shot Dense Retrieval without Relevance Labels" by Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan (2022).

HyDE addresses the fundamental asymmetry between short queries and long documents in dense retrieval by generating a hypothetical document that the query might retrieve, then using that document's embedding for retrieval instead of the query's embedding.

### 2.2 Step-by-Step Algorithm

1. **Query Reception:** User query `q` is received
2. **Hypothetical Document Generation:** An instruction-following LLM (e.g., InstructGPT) generates a hypothetical document `d_hat` that would answer the query. This document need NOT be factually accurate.
3. **Document Encoding:** An unsupervised contrastive encoder (e.g., Contriever) encodes `d_hat` into an embedding vector `v_hat`
4. **Similarity Search:** `v_hat` is compared against the corpus embeddings `{v_i}` using inner product similarity
5. **Document Retrieval:** Top-k real documents with highest similarity to `v_hat` are retrieved
6. **Response Generation:** Retrieved real documents are used to generate the final answer

### 2.3 Mathematical Formulation

**Basic query vector from hypothetical document:**
```
v_q = f_e(d_hat)    where d_hat = G(q), G is the LLM, f_e is the encoder
```

**Multiple hypothesis averaging (N hypotheses):**
```
v_q = (1/N) * SUM(f_e(d_hat_k) for k=1..N)
```

**With original query blending:**
```
v_q = (1/(N+1)) * (SUM(f_e(d_hat_k)) + f_e(q))
```

**Similarity matching:**
```
similarity(q, d) = dot(v_q, v_d)
```

**Rocchio Feedback Extension:**
```
w_{t, q_new} = alpha * f(q)[t] + (beta/N) * SUM(f_tilde(d_hat_i)[t])
```

**Source:** [HyDE Paper](https://arxiv.org/abs/2212.10496)

### 2.4 Multiple Hypothesis Averaging

Generating N hypothetical documents and averaging their embeddings:
- Reduces variance from any single hallucinated hypothesis
- The encoder's dense bottleneck filters out incorrect details
- Captures relevance patterns while smoothing errors
- Typical N = 4-8 hypotheses in practice

### 2.5 Implementation in LangChain

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Basic HyDE setup
llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"  # or "qa", "sci_fact"
)

# Use as drop-in replacement for regular embeddings
vectorstore = Chroma.from_documents(documents, hyde_embeddings)
```

### 2.6 Implementation in LlamaIndex

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(
    query_engine=base_query_engine,
    query_transform=hyde
)
```

### 2.7 Prompt Templates for Different Domains

**General / Web Search:**
```
Please write a passage to answer the question.
Question: {query}
Passage:
```

**Medical Domain:**
```
As a board-certified [specialist], write a detailed clinical passage
that answers the following medical question.
Question: {query}
Passage:
```

**Technical / QA:**
```
For the given question, try to generate a hypothetical answer.
Only generate the answer and nothing else.
Question: {question}
```

**Fact Verification:**
```
Please write a scientific passage that would support or refute
the following claim.
Claim: {query}
Passage:
```

**Source:** [HyDE for RAG Explained](https://machinelearningplus.com/gen-ai/hypothetical-document-embedding-hyde-a-smarter-rag-method-to-search-documents/)

### 2.8 Benchmarks on Different Datasets

| Method | Dataset | Metric | Score |
|--------|---------|--------|-------|
| HyDE | TREC DL-20 | nDCG@10 | 61.3 |
| Contriever (baseline) | TREC DL-20 | nDCG@10 | 44.5 |
| HyDE | BEIR (avg) | nDCG@10 | Surpasses Contriever zero-shot |
| HyDE + Rocchio | MS MARCO/TREC | Recall@20 | +4.2% absolute over naive |
| SL-HyDE | Medical | nDCG@10 | 59.38% vs 56.62% vanilla HyDE |
| HyDE sparse variant | BEIR (avg) | nDCG@10 | +6.0% average over baselines |
| Adaptive HyDE | Developer QA | Helpfulness | +20% over standard RAG |

**Key finding:** HyDE significantly outperforms the state-of-the-art unsupervised dense retriever Contriever and shows strong performance comparable to fine-tuned retrievers across various tasks and languages.

**Sources:**
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [HyDE Topic Overview - Emergent Mind](https://www.emergentmind.com/topics/hypothetical-document-embeddings-hyde)

### 2.9 When HyDE Hurts Performance

| Scenario | Why It Fails |
|----------|-------------|
| **Factoid/keyword queries** | Short, specific queries already match well; HyDE adds noise |
| **Highly specialized domains** | LLM lacks domain knowledge, generates poor hypotheses |
| **Personal/private data queries** | LLM has no training data about user-specific content; high hallucination rate |
| **Novel/emerging topics** | Topics not in LLM training data produce inaccurate hypotheses |
| **Multi-part constrained queries** | Complex constraints generate irrelevant hypothetical documents |
| **Real-time applications** | 1-5 second generation overhead is unacceptable |
| **High-throughput systems** | Queue congestion during peak usage; timeout risks |
| **Temperature sensitivity** | Model temperature variations produce inconsistent results for identical queries |

**Quantified degradation:**
- On small LLMs (Gemma 1B/4B): HyDE incurs 25-60% increase in latency over standard RAG
- For personal data retrieval: Non-negligible hallucination rate
- HyDE improves semantic alignment but NOT recall in many cases

**Sources:**
- [Inverted HyDE - Behitek](https://behitek.com/blog/inverted-hyde/)
- [HyDE Limitations - Milvus](https://milvus.io/ai-quick-reference/what-is-hyde-hypothetical-document-embeddings-and-when-should-i-use-it)
- [HyDE vs RAG - BeyondScale](https://beyondscale.tech/blog/hyde-vs-rag-retrieval-augmented-generation)

### 2.10 Latency and Cost Analysis

| Component | Added Latency | Added Cost |
|-----------|---------------|------------|
| Hypothesis generation (1 doc) | 500ms - 5s | 1 LLM call (~100-500 tokens output) |
| Hypothesis generation (N=4) | 2-10s | 4 LLM calls (or 1 batched) |
| Encoding hypothetical doc | 10-50ms | 1 embedding call |
| Total overhead vs direct query | **1-5 seconds minimum** | ~$0.001-0.01 per query |

**Inverted HyDE Alternative:** Generate hypothetical queries for each document at indexing time (offline), eliminating query-time LLM dependency entirely. Converts document-to-query matching into query-to-query similarity space.

**Source:** [Inverted HyDE](https://behitek.com/blog/inverted-hyde/)

---

## 3. Query Routing

### 3.1 LLM-Based Semantic Routing

Semantic routing uses embeddings and cosine similarity to match incoming queries against predefined route descriptions. The router:

1. Encodes the user query into an embedding
2. Compares against pre-embedded route descriptions/utterances
3. Routes to the highest-similarity route above a confidence threshold
4. Falls back to a default route if no match exceeds threshold

**Speed advantage:** Reduces routing latency from ~5000ms (LLM-based) to ~100ms (embedding-based).

### 3.2 Keyword-Based Routing

- Pattern matching on query terms (regex, keyword lists)
- Fast and deterministic
- Limited to explicit mentions; misses semantic intent
- Best for clear-cut routing (e.g., SQL keywords -> database retrieval)

### 3.3 Metadata-Based Routing

- Uses query metadata (user role, department, language, timestamp)
- Routes based on known context rather than query content
- Useful for multi-tenant systems with segmented knowledge bases

### 3.4 Multi-Index Routing

Different vector stores or indices for different content types:
- Technical documentation -> dense vector index
- API references -> keyword/BM25 index
- FAQ -> small, curated vector index
- Structured data -> SQL/graph database

### 3.5 LangChain Routing Implementations

**EmbeddingRouterChain:**
```python
from langchain.chains.router import EmbeddingRouterChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

# Define routes with sample queries
route_templates = {
    "technical": ["How do I configure the API?", "What's the error code for..."],
    "billing": ["What's my invoice total?", "How do I update payment..."],
    "general": ["What does your company do?", "Tell me about..."]
}

# Embed route examples into a lightweight vector store
names_and_descriptions = [
    ("technical", ["How do I configure...", "API documentation..."]),
    ("billing", ["Invoice questions...", "Payment methods..."]),
]

router_chain = EmbeddingRouterChain.from_names_and_descriptions(
    names_and_descriptions,
    DocArrayInMemorySearch,
    OpenAIEmbeddings(),
    routing_keys=["input"]
)
```

**LLM-Based Router:**
```python
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

router_template = """Given the user question below, classify it into one of the
following categories: {destinations}

<question>
{input}
</question>

Classification:"""

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
```

**Sources:**
- [LangChain Embedding Router](https://python.langchain.com/docs/expression_language/cookbook/embedding_router)
- [Routing in RAG Applications - TDS](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220/)

### 3.6 LlamaIndex RouterQueryEngine

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, PydanticSingleSelector

# Define query engine tools
query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for specific fact-based questions"
    ),
    QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions"
    ),
]

# Router with LLM selector
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
)
```

**Selector Types:**
| Selector | Description |
|----------|-------------|
| `LLMSingleSelector` | Selects ONE choice via text completion |
| `LLMMultiSelector` | Selects MULTIPLE choices via text completion |
| `PydanticSingleSelector` | Uses function calling with Pydantic schemas |
| `PydanticMultiSelector` | Returns multiple Pydantic objects |

**Source:** [LlamaIndex Router Documentation](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/)

### 3.7 Semantic Router Library (Aurelio AI)

```python
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder

# Define routes
technical = Route(
    name="technical",
    utterances=[
        "How do I configure the API?",
        "What's the authentication method?",
        "Show me the code example",
    ]
)

billing = Route(
    name="billing",
    utterances=[
        "What's my invoice?",
        "How do I update payment?",
    ]
)

encoder = OpenAIEncoder()
route_layer = RouteLayer(encoder=encoder, routes=[technical, billing])

# Route a query
result = route_layer("How do I set up OAuth?")  # -> "technical"
```

**Features:**
- Embedding flexibility (Cohere, OpenAI, Hugging Face encoders)
- Latency: ~100ms (vs 5000ms for LLM-based routing)
- Scalable to thousands of routes
- Threshold management for false positive reduction
- Works with Pinecone, Qdrant, or in-memory stores

**Sources:**
- [Semantic Router GitHub](https://github.com/aurelio-labs/semantic-router)
- [Semantic Router - Deepchecks](https://www.deepchecks.com/glossary/semantic-router/)
- [SemanticRouter - RedisVL](https://docs.redisvl.com/en/latest/user_guide/08_semantic_router.html)

### 3.8 Confidence Thresholds and Fallback Strategies

- **Distance threshold:** Maximum cosine distance for route matching (typical: 0.3-0.5)
- **If below threshold:** Route to default/general-purpose retriever
- **Multi-route fan-out:** If multiple routes score above threshold, query all and merge results
- **Cascading fallback:** Try semantic routing -> keyword routing -> full corpus search

### 3.9 Production Patterns

- Use lightweight in-memory vector store for routes (DocArrayInMemorySearch) -- typically <100 route utterances
- Cache route decisions for repeated queries
- Monitor misroute rates and add utterances to underperforming routes
- Keep route descriptions concise and distinct
- Use organization-specific terminology in route utterances for private chatbots

---

## 4. Query Decomposition

### 4.1 Sub-Question Decomposition

**Algorithm:**
1. Complex query is analyzed by an LLM
2. LLM breaks it into 2-4 independent sub-questions
3. Each sub-question is processed through retrieval independently
4. Retrieved contexts are aggregated
5. Final synthesis produces a comprehensive answer

**Implementation:**
```python
subquery_decomposition_template = """You are an AI assistant tasked with breaking down
complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered
together, would provide a comprehensive response to the original query.

Original query: {original_query}

Sub-queries:"""

def decompose_query(original_query: str):
    response = chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip()]
    return sub_queries
```

**Example:**
- Original: "Did Microsoft or Google make more money last year?"
- Sub-query 1: "What was Microsoft's revenue last year?"
- Sub-query 2: "What was Google's revenue last year?"

**Source:** [Haystack Query Decomposition](https://haystack.deepset.ai/blog/query-decomposition)

### 4.2 Multi-Query Generation vs Rewriting

| Aspect | Multi-Query Rewriting | Query Decomposition |
|--------|----------------------|---------------------|
| **Goal** | Cover same intent from multiple angles | Break complex query into independent parts |
| **Output** | Paraphrases of same question | Different, simpler questions |
| **Retrieval** | Parallel retrieval, merge results | Independent retrieval per sub-query |
| **Aggregation** | Union + dedup | Synthesis across sub-answers |
| **Best for** | Ambiguous queries | Multi-hop reasoning queries |

### 4.3 Step-Back Prompting

**Paper:** "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models" (arXiv:2310.06117) by Google DeepMind.

**Algorithm:**
1. Instead of answering directly, ask a broader/more abstract question first
2. Retrieve background information using the step-back question
3. Use retrieved context + original question for final reasoning

**Implementation:**
```python
step_back_template = """You are an AI assistant tasked with generating broader, more
general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help
retrieve relevant background information.

Original query: {original_query}

Step-back query:"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)
```

**Example:**
- Original: "How do I fix this React useEffect infinite loop?"
- Step-back: "What are common causes of infinite re-render loops in React?"

**Benchmark Results (PaLM-2L):**
| Task | Improvement |
|------|-------------|
| MMLU Physics | +7% |
| MMLU Chemistry | +11% |
| TimeQA | +27% |
| MuSiQue | +7% |
| Step-Back + RAG vs baseline on TimeQA | Fixed 39.9% of incorrect predictions; only 5.6% new errors |

Step-Back + RAG produced better results than either technique alone. Up to 36% improvement over chain-of-thought prompting.

**Sources:**
- [Step-Back Prompting Paper](https://arxiv.org/abs/2310.06117)
- [PromptHub Blog](https://www.prompthub.us/blog/a-step-forward-with-step-back-prompting)
- [The Decoder](https://the-decoder.com/deepminds-new-prompting-method-takes-a-step-back-for-more-accuracy/)

### 4.4 Least-to-Most Prompting for Complex Queries

Least-to-Most Prompting decomposes a challenging task into simpler subproblems solved sequentially, where solutions to earlier subproblems feed into prompts for later ones.

**Key difference from Chain-of-Thought:** Solutions are cumulative -- each step builds on the previous.

**Application in RAG:** Break a complex retrieval query into progressively more specific sub-queries, using early retrievals to inform later queries.

**Source:** [Least-to-Most Prompting - Learn Prompting](https://learnprompting.org/docs/intermediate/least_to_most)

### 4.5 Tree of Thought for Query Planning

Tree of Thoughts (ToT) maintains a tree of reasoning paths where:
- Each node is a coherent language sequence (intermediate reasoning step)
- Search algorithms (BFS/DFS) explore the tree systematically
- LLM self-evaluates progress at each node
- Enables lookahead and backtracking

**Application in RAG:** Complex multi-hop questions are decomposed into a tree structure with entity analysis, bottom-up traversal with query refinement, and hierarchical information integration (RT-RAG approach).

**Source:** [Tree of Thoughts - Prompt Engineering Guide](https://www.promptingguide.ai/techniques/tot)

### 4.6 LangChain and LlamaIndex Implementations

**LangChain:**
```python
# Decomposition
from langchain.retrievers import DecomposingRetriever
retriever = DecomposingRetriever(llm=llm, retriever=base_retriever)

# Step-back
from langchain.retrievers import StepBackRetriever
retriever = StepBackRetriever(
    llm=llm, retriever=base_retriever,
    step_back_template=STEP_BACK_PROMPT
)

# Multi-query
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    llm=llm, retriever=vectorstore.as_retriever(), num_queries=3
)
```

**LlamaIndex:**
```python
from llama_index.core.query_engine import SubQuestionQueryEngine

engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)
```

**Haystack (with Pydantic structured output):**
```python
from pydantic import BaseModel
from typing import List

class Question(BaseModel):
    question: str

class Questions(BaseModel):
    questions: List[Question]

# Uses OpenAI structured output to generate sub-questions
```

### 4.7 Aggregating Results from Sub-Queries

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Union + Dedup** | Merge all docs, remove duplicates by content hash | Multi-query rewriting |
| **Weighted Vote** | Score documents by how many sub-queries retrieved them | Consensus-based retrieval |
| **Sequential Synthesis** | Answer sub-queries in order, feed into next | Least-to-most reasoning |
| **Map-Reduce** | Answer each sub-query independently, then reduce | Independent sub-questions |
| **Reranking** | Pool all results, rerank with cross-encoder | Any multi-query approach |

**Deduplication strategies:**
- Content hash deduplication (keep highest similarity score)
- Semantic deduplication (cluster similar docs, keep representative)
- Score-weighted fusion (Reciprocal Rank Fusion)

### 4.8 When Decomposition Hurts (Over-Decomposition)

- **Simple factoid queries:** Decomposition adds latency without benefit
- **Overly granular sub-questions:** Lose the holistic context of the original query
- **Independent sub-queries that need joint context:** Decomposing a comparative query into parts that each lack the comparison aspect
- **Latency-sensitive applications:** Each sub-query adds a retrieval round-trip
- **Small knowledge bases:** Multiple queries against a small corpus return the same documents

**DecomposeRAG** (2025): A framework that automatically breaks complex queries into simpler sub-queries, achieving state-of-the-art results on multi-hop QA benchmarks, handling complex questions 50% better.

**Source:** [DecomposeRAG Research](https://app.ailog.fr/en/blog/news/query-decomposition-research)

---

## 5. Query Classification / Intent Detection

### 5.1 Classification Categories

| Intent Type | Description | Retrieval Strategy |
|-------------|-------------|-------------------|
| **Factoid** | Specific fact lookups | Dense vector retrieval, high precision |
| **Analytical** | Deeper explanations/comparisons | Multiple chunks, broader retrieval |
| **Conversational** | Chitchat, clarification | Minimal/no retrieval needed |
| **Opinion** | Subjective viewpoints | Diverse document retrieval |
| **Contextual** | Relies on conversation history | Context-aware retrieval |
| **Summarization** | Condensing information | Full-document retrieval |
| **Navigational** | Finding specific pages/resources | Keyword/metadata search |

### 5.2 Implementation Approaches

**a) LLM-Based Classification:**
```python
classification_prompt = """Classify the following query into one of these categories:
- factoid: specific fact lookup
- analytical: requires analysis or comparison
- conversational: chitchat or clarification
- navigational: looking for a specific resource

Query: {query}

Category:"""
```

**b) Supervised Classifier (Cheaper):**
- Train a small BERT/DistilBERT classifier on labeled query data
- Inference: <10ms, cost: negligible
- Accuracy: 85-95% with good training data

**c) Zero-Shot Classification:**
- Use models like `facebook/bart-large-mnli` for zero-shot intent classification
- No task-specific training needed
- Moderate accuracy, no API cost (if self-hosted)

**d) Rule-Based Classification:**
- Query length, question words, domain keywords
- Fast and deterministic
- Limited but useful as a first filter

### 5.3 REIC: RAG-Enhanced Intent Classification

Amazon's REIC framework uses RAG itself to enhance intent classification accuracy:
- **Index construction:** Build index of intent examples
- **Candidate retrieval:** Retrieve similar past queries with known intents
- **Intent probability calculation:** Score intent based on retrieved examples

Key finding: Among dense retrievers, **MPNet outperforms others** for intent classification retrieval. **BM25 performs competitively** despite being unsupervised.

**Source:** [REIC Paper](https://arxiv.org/html/2506.00210v1)

### 5.4 How Classification Feeds into Routing

```
Query -> Classifier -> {
    factoid     -> Direct vector retrieval (top-3, high threshold)
    analytical  -> Multi-query expansion + broader retrieval (top-10)
    conversational -> Skip retrieval, direct LLM response
    navigational -> Keyword/BM25 search
    complex     -> Query decomposition + multi-index routing
}
```

### 5.5 Using Smaller Models for Cost Optimization

| Model | Latency | Cost | Accuracy |
|-------|---------|------|----------|
| GPT-4 | 500-2000ms | $$$ | ~95% |
| GPT-4o-mini | 100-500ms | $ | ~90% |
| Fine-tuned DistilBERT | 5-10ms | ~$0 | ~90% |
| Rule-based | <1ms | $0 | ~70-80% |

**Recommendation:** Use a fine-tuned small classifier (DistilBERT/TinyBERT) for classification, reserve LLM calls for actual rewriting/generation.

**Sources:**
- [Intent Classification in RAG - Medium](https://alixaprodev.medium.com/how-intent-classification-works-in-rag-systems-15054d0ec5ce)
- [Adaptive RAG Strategy - n8n](https://n8n.io/workflows/3459-adaptive-rag-strategy-with-query-classification-and-retrieval-gemini-and-qdrant/)
- [HitReader - Improve RAG](https://www.hitreader.com/improve-rag/)

---

## 6. Query Expansion with Pseudo-Relevance Feedback (PRF)

### 6.1 Traditional PRF Adapted for RAG

**Classic PRF Process:**
1. **Initial retrieval:** User query returns top-N documents
2. **Assumption:** Top-N documents are relevant (the "pseudo" part)
3. **Term extraction:** Extract significant terms from top-N documents
4. **Term weighting:** Weight each term by importance/relevance score
5. **Query refinement:** Expand original query with weighted terms
6. **Second retrieval:** Run expanded query for final results

**Adaptation for RAG:** Instead of traditional term extraction, use the retrieved chunks to inform a second, more targeted retrieval pass.

### 6.2 RM3 (Relevance-Based Language Model)

RM3 is an association-based query expansion method that:
- Uses co-occurrence information between candidate terms and original query terms
- Interpolates the original query language model with a relevance model estimated from pseudo-relevant documents
- Controlled by parameters: number of feedback documents, number of expansion terms, interpolation weight

**Formula:**
```
P(w|Q') = alpha * P(w|Q) + (1-alpha) * P(w|R)
```
Where R is the set of pseudo-relevant documents, alpha controls interpolation.

### 6.3 KL-Divergence Based Expansion

- Compares the distribution of a term in pseudo-relevant documents vs the whole corpus
- Terms with high KL-divergence (i.e., overrepresented in relevant docs) are good expansion candidates
- Distribution-based, making it complementary to association-based RM3

**Bo1 (Bose-Einstein):** Another distribution-based method that models term frequency using Bose-Einstein statistics.

**Source:** [PyTerrier Query Rewriting](https://pyterrier.readthedocs.io/en/latest/rewrite.html)

### 6.4 Neural PRF Approaches

| Method | Description |
|--------|-------------|
| **ANCE-PRF** | Fine-tunes ANCE retriever with pseudo-relevant feedback |
| **ColBERT-PRF** | Extends ColBERT with PRF-based query expansion |
| **CEQE** | Replaces frequency counts in RM3 with context-sensitive similarities |
| **GRF (Generative Relevance Feedback)** | Replaces the relevant document set with LLM-produced set, estimates relevance model using RM3-style term distribution with language model likelihoods |

**Key insight:** Vector-based PRF enhances effectiveness of deep rerankers and dense retrievers. Higher effectiveness when the query retains majority weight within PRF mechanism, and shallower PRF signal is employed.

**Source:** [Pseudo Relevance Feedback with Deep Language Models](https://arxiv.org/abs/2108.11044)

### 6.5 When PRF Improves vs Hurts Performance

| Scenario | PRF Effect |
|----------|-----------|
| **Broad, underspecified queries** | IMPROVES - adds relevant context terms |
| **Well-specified queries** | NEUTRAL to SLIGHTLY HURTS - may add noise |
| **Topic drift in top docs** | HURTS - learns incorrect patterns from irrelevant top docs |
| **Niche/specialized domains** | RISKY - initial retrieval may be poor |
| **High-quality initial retrieval** | DIMINISHING RETURNS |

**Challenges for neural search:**
- PRF methods haven't widely penetrated neural/vector search systems
- The cost-efficiency balance hasn't been found for production neural PRF
- Most implementations remain in research, not production

**Sources:**
- [Relevance Feedback in IR - Qdrant](https://qdrant.tech/articles/search-feedback-loop/)
- [PRF and LLM Techniques - Medium](https://medium.com/learnwithnk/the-mechanics-of-query-expansion-in-rag-systems-a-theoretical-exploration-of-prf-and-llm-6e66327ad300)
- [Query Expansion Survey](https://arxiv.org/pdf/2509.07794)

---

## 7. Combining Techniques

### 7.1 Technique Combination Matrix

| Combination | When to Use | Trade-off |
|-------------|-------------|-----------|
| **HyDE + Multi-Query** | Maximum recall for ambiguous queries | High latency, high cost |
| **Classification + Routing** | Multi-index systems with diverse content | Minimal overhead, big impact |
| **Step-Back + Decomposition** | Complex multi-hop reasoning | Multiple LLM calls |
| **Rewriting + Routing** | Enterprise systems with many data sources | Moderate overhead |
| **HyDE + Routing** | Short queries + multi-index | Medium latency |
| **Classification + HyDE (conditional)** | Only use HyDE when classifier detects underspecified queries | Smart cost/latency management |

### 7.2 Adaptive Strategy Selection

**Production decision tree:**
```
Query arrives ->
  1. Classify query (fast classifier, <10ms)
  2. Route based on classification:
     - Simple factoid -> Direct embedding, no transformation
     - Ambiguous/short -> HyDE or query expansion
     - Complex multi-part -> Decomposition
     - Analytical -> Multi-query + step-back
     - Conversational -> Skip retrieval
  3. Apply transformation(s)
  4. Retrieve from appropriate index
  5. Rerank with cross-encoder
  6. Generate response
```

**Dynamic selection from DMQR-RAG:**
The adaptive strategy selection method minimizes the number of rewrites while optimizing overall performance, choosing the best rewriting strategy per query.

### 7.3 Mature Production Pipeline

```
If query is very short         -> HyDE
Else if ambiguity score is high -> Multi-Query RAG
Else                           -> Query Expansion (keyword-based)
```

- Cache HyDE outputs aggressively
- Cap Multi-Query fanout at 3-5 queries
- Rerank with cross-encoders or LLMs
- Track Recall@K, MRR, and latency budgets

**Source:** [Retrieval Is the Bottleneck - Medium](https://medium.com/@mudassar.hakim/retrieval-is-the-bottleneck-hyde-query-expansion-and-multi-query-rag-explained-for-production-c1842bed7f8a)

---

## 8. Production Deployment Patterns

### 8.1 Latency Budget Allocation

**Target:** 2-5 seconds for complex queries, 100-1000 queries per minute throughput.

| Stage | Latency Budget | Notes |
|-------|---------------|-------|
| Query classification | 5-10ms | Fine-tuned small model |
| Query routing | 10-100ms | Embedding similarity |
| Query rewriting/HyDE | 200ms-2s | LLM call (use fast model) |
| Embedding generation | 10-50ms | Per query variant |
| Vector search | 10-100ms | Per query variant |
| Reranking | 50-200ms | Cross-encoder |
| **Total pre-retrieval** | **~300ms-2.5s** | Depends on technique |

### 8.2 Cost Optimization Strategies

1. **Model tiering:** Use GPT-4o-mini or Haiku for rewriting; reserve GPT-4/Claude for generation
2. **Conditional transformation:** Only apply HyDE/multi-query when classifier detects need
3. **Caching:** Cache rewrite results for repeated/similar queries (prompt caching saves up to 85% latency)
4. **Batch processing:** Batch embed multiple query variants in a single API call
5. **Pre-computation:** Inverted HyDE (generate hypothetical queries at indexing time)
6. **Simple queries route to cheaper models** reducing LLM costs by 50%+
7. **Context optimization** can reduce LLM costs by 40-60%

**Sources:**
- [Production RAG Pipelines - HackerNoon](https://hackernoon.com/designing-production-ready-rag-pipelines-tackling-latency-hallucinations-and-cost-at-scale)
- [RAG Cost Optimization](https://zenvanriel.com/ai-engineer-blog/rag-cost-optimization-strategies/)
- [Microsoft RAG Optimization](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/from-zero-to-hero-proven-methods-to-optimize-rag-for-production/4450040)

### 8.3 Infrastructure Patterns

- **Distributed vector databases** with sharding for scalable, low-latency retrieval
- **GPU-accelerated models** for embedding and reranking
- **Asynchronous orchestration** for parallel query variant processing
- **Hybrid retrieval** (BM25 + dense) achieves 50% latency reduction
- **Pre-computed embeddings** for all documents at indexing time

### 8.4 Monitoring and Observability

Key metrics to track:
- **Retrieval quality:** Recall@K, Precision@K, MRR, nDCG@K
- **Latency:** P50, P95, P99 per pipeline stage
- **Cost:** Cost per satisfactory response
- **Routing accuracy:** Misroute rate, fallback rate
- **Query transformation quality:** Semantic similarity between original and rewritten queries

---

## 9. Evaluation and A/B Testing

### 9.1 Evaluating Pre-Retrieval Quality Independently

**Retrieval-only metrics (independent of generation):**

| Metric | What It Measures |
|--------|-----------------|
| **Precision@K** | % of retrieved docs that are relevant |
| **Recall@K** | % of relevant docs in corpus that were retrieved |
| **F1@K** | Harmonic mean of precision and recall |
| **MRR (Mean Reciprocal Rank)** | Position of first relevant result |
| **nDCG@K** | Ranking quality with graded relevance |
| **Hit Rate** | Whether ANY relevant doc appears in top-K |

**Evaluating query transformation specifically:**
1. Compare Recall@K of original query vs transformed query against same ground truth
2. Measure semantic similarity between original and rewritten query (should be high)
3. Track diversity of results across multi-query variants
4. Monitor over-retrieval rate (too many irrelevant docs)

**Precision-recall tradeoff guidance:**
- Fewer documents (3-5): Higher precision, lower cost, risk of missing information
- More documents (10-20): Higher recall, more noise, higher latency

### 9.2 A/B Testing Strategies

**Pipeline Comparison:**
- Present new pipeline (with query rewriting) to subset of users
- Others experience baseline pipeline
- Compare on quality metrics + latency + cost

**Dual Pipeline Evaluation:**
- Run both pipelines on same queries simultaneously
- Use LLM-as-judge to evaluate which produced better results
- No user exposure needed for initial comparison

**Multi-Armed Bandit:**
- Dynamically allocate more traffic to better-performing variants
- Minimizes regret during experimentation
- Well-suited for optimizing LLM prompts and rewriting strategies

**Interleaving Experiments:**
- Present mixed results from multiple pipelines in a single result set
- Use clicks/engagement as evidence of which pipeline produces better results
- More statistically powerful than traditional A/B tests

### 9.3 Component-Level Testing

Isolate and test each component independently:
1. **Rewriting quality:** Compare retrieval metrics with/without rewriting
2. **Routing accuracy:** Measure classification accuracy against labeled test set
3. **Decomposition effectiveness:** Compare on multi-hop QA benchmarks
4. **HyDE vs direct:** Compare retrieval quality on query subsets by type

**Methodology:** Modify one component at a time, re-run evaluation, measure the delta. This shows each component's individual contribution.

**Sources:**
- [RAG Evaluation Guide - Evidently AI](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Retrieval Quality Evaluation - TDS](https://towardsdatascience.com/how-to-evaluate-retrieval-quality-in-rag-pipelines-precisionk-recallk-and-f1k/)
- [RAG Evaluation - Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)
- [A/B Testing for RAG - Dataworkz](https://www.dataworkz.com/blog/a-b-testing-strategies-for-optimizing-rag-applications/)
- [RAG Evaluation Best Practices - Qdrant](https://qdrant.tech/blog/rag-evaluation-guide/)

---

## Summary Comparison Table

| Technique | Added Latency | Added Cost | Best For | Improvement Range |
|-----------|--------------|------------|----------|-------------------|
| **Query Rewriting** | 200-700ms | 1 LLM call | Ambiguous/conversational queries | +25-50% precision |
| **Multi-Query** | 300-900ms | 1 LLM call + N embeddings | Vocabulary mismatch | +15-35% recall |
| **HyDE** | 1-5s | 1 LLM call + 1 embedding | Short, underspecified queries | Surpasses Contriever; nDCG@10 up to 61.3 vs 44.5 |
| **Query Routing** | 10-100ms | Negligible (embedding) | Multi-index systems | Avoids misrouting; precision per-index |
| **Query Decomposition** | 500-2000ms | 1 LLM call + N retrievals | Complex multi-hop queries | +50% on complex QA (DecomposeRAG) |
| **Step-Back Prompting** | 300-1000ms | 1 LLM call | Questions requiring background | +7-27% on reasoning tasks |
| **Intent Classification** | 5-100ms | Negligible | Strategy selection | Enables all above conditionally |
| **PRF** | 200-500ms | 1 extra retrieval | Broad, underspecified queries | Variable; risk of topic drift |

---

## Key Research Papers

1. **HyDE:** Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) - [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
2. **Step-Back Prompting:** Zheng et al., "Take a Step Back: Evoking Reasoning via Abstraction in LLMs" (2023) - [arXiv:2310.06117](https://arxiv.org/abs/2310.06117)
3. **DMQR-RAG:** "Diverse Multi-Query Rewriting for RAG" (2024) - [arXiv:2411.13154](https://arxiv.org/abs/2411.13154)
4. **DecomposeRAG:** Query Decomposition for Complex Questions (2025) - [Coverage](https://app.ailog.fr/en/blog/news/query-decomposition-research)
5. **REIC:** "RAG-Enhanced Intent Classification at Scale" (2025) - [arXiv:2506.00210](https://arxiv.org/html/2506.00210v1)
6. **Adaptive HyDE:** "Never Come Up Empty" (2025) - [arXiv:2507.16754](https://arxiv.org/abs/2507.16754)
7. **Neural PRF:** "Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers" - [arXiv:2108.11044](https://arxiv.org/abs/2108.11044)
8. **CoT-RAG:** "Integrating Chain of Thought and Retrieval" (2025) - [ACL Findings](https://aclanthology.org/2025.findings-emnlp.168.pdf)
9. **Query Expansion Survey:** "Query Expansion in the Age of Pre-trained and Large Language Models" - [arXiv:2509.07794](https://arxiv.org/pdf/2509.07794)
