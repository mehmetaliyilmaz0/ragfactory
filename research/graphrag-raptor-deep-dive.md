# GraphRAG & RAPTOR: Exhaustive Architecture Research

> Research compiled: 2026-04-04
> Sources: Microsoft Research, Stanford NLP, arXiv, Neo4j, LlamaIndex, and community benchmarks

---

## Table of Contents

- [Part 1: GraphRAG (Microsoft)](#part-1-graphrag-microsoft)
  - [1.1 Architecture Deep Dive](#11-architecture-deep-dive)
  - [1.2 Query Modes Detailed](#12-query-modes-detailed)
  - [1.3 Configuration Deep Dive](#13-configuration-deep-dive)
  - [1.4 Performance & Benchmarks](#14-performance--benchmarks)
  - [1.5 Implementation](#15-implementation)
  - [1.6 Limitations and Edge Cases](#16-limitations-and-edge-cases)
- [Part 2: RAPTOR](#part-2-raptor)
  - [2.1 Algorithm Deep Dive](#21-algorithm-deep-dive)
  - [2.2 Configuration Parameters](#22-configuration-parameters)
  - [2.3 Benchmarks](#23-benchmarks)
  - [2.4 Implementation](#24-implementation)
  - [2.5 Limitations](#25-limitations)
- [Part 3: Other Graph-Based and Hierarchical Approaches](#part-3-other-graph-based-and-hierarchical-approaches)
- [Part 4: Decision Matrix](#part-4-decision-matrix)

---

# Part 1: GraphRAG (Microsoft)

**Repository**: [microsoft/graphrag](https://github.com/microsoft/graphrag)
**Paper**: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research, April 2024)
**Docs**: [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/)

## 1.1 Architecture Deep Dive

### Full Indexing Pipeline

The GraphRAG indexing pipeline transforms raw documents into a structured knowledge model through six sequential phases:

#### Phase 1: TextUnit Composition
- Raw documents (CSV rows or .txt files) are chunked into **text units**
- Default chunk size: **1,200 tokens** per chunk (configurable)
- Configurable overlap between chunks
- Sentence boundaries are preserved for coherence
- Each text unit maintains a reference back to its source document

#### Phase 2: Document-TextUnit Linking
- Links each document to its constituent text units
- Creates provenance tracking back to source materials
- Outputs a Documents table in the knowledge model

#### Phase 3: Graph Extraction
Three sub-processes run on each text unit:

1. **Entity & Relationship Extraction**: An LLM extracts named entities (people, places, organizations, concepts) and the relationships between them from each text unit. The LLM is prompted with configurable entity types and provides descriptions for each entity and relationship.

2. **Entity & Relationship Summarization**: Duplicate entities across text units are merged. The LLM generates consolidated descriptions for entities that appear multiple times with potentially different context.

3. **Claim Extraction** (optional, off by default): Extracts factual statements with status indicators (true/false/unknown) and time-bound metadata.

**Two extraction methods available:**
- **LLM-based (standard)**: Uses the language model for entity/relationship extraction -- higher quality, higher cost
- **NLP-based (FastGraphRAG)**: Uses spaCy/NLTK noun-phrase extraction -- faster, cheaper, lower quality; entities are noun phrases extracted via traditional NLP

**Max Gleanings**: After the initial extraction pass, the system can perform additional "gleaning" passes (configurable via `max_gleanings`) to catch entities and relationships missed in the first pass. Each gleaning pass adds cost but improves recall.

#### Phase 4: Graph Augmentation (Community Detection)
- The Hierarchical Leiden Algorithm is applied to the entity-relationship graph
- Produces recursive community clusters organized in a hierarchy
- Outputs a Communities table with hierarchical organization

#### Phase 5: Community Summarization
- An LLM generates a report for each community, describing its key entities, relationships, and themes
- Shorthand summary versions are also created
- Outputs Community Reports table

#### Phase 6: Text Embedding
- Generates vector embeddings for:
  - Entity descriptions
  - Text unit content
  - Community report full content
- Embeddings stored in configured vector store (LanceDB default, Azure AI Search, CosmosDB supported)

### Leiden Algorithm for Community Detection

The Leiden algorithm is a state-of-the-art community detection method that improves upon the Louvain algorithm by adding a refinement phase.

**How it works:**
1. **Local Moving Phase**: Each node is assigned to the community that maximizes modularity gain
2. **Refinement Phase**: Verifies each community is internally well-connected, ensuring no isolated nodes remain within a cluster (key improvement over Louvain)
3. **Aggregation Phase**: Communities are aggregated into single nodes, creating a condensed graph
4. **Recursion**: Steps 1-3 repeat on the condensed graph until no further improvement

**Hierarchy Levels:**
- **Level 0**: Original graph, finest-grained communities detected
- **Level 1**: Aggregates Level 0 communities into larger groups
- **Level 2+**: Continues aggregating until no further improvement
- Number of communities decreases at each level
- The number of levels depends on graph structure (typically 3-6 for real datasets)

**Key Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_cluster_size` | Maximum nodes per community for export | Varies |
| `use_lcc` | Only use largest connected component | False |
| `seed` | Random seed for reproducibility | None |
| `resolution` (gamma) | Higher = smaller, more specific communities; Lower = broader clusters | 1.0 |
| `randomness` | Controls exploration of solution space | Varies |
| `iterations` | Number of optimization iterations | Varies |

### Entity Extraction Prompts and Customization

- Default prompts extract entities of types: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT
- Entity types are fully configurable via `entity_types` list in YAML config
- Custom prompt templates can be provided via `extract_graph.prompt` file path
- The prompt instructs the LLM to output structured JSON with entity name, type, description, and relationships
- **Domain-specific tuning**: For specialized corpora (medical, legal, scientific), custom entity types and extraction prompts dramatically improve quality

### Relationship Extraction and Weighting

- Relationships are extracted as triples: (source_entity, relationship_description, target_entity)
- Each relationship includes a natural language description
- Relationship weights can be normalized during graph construction (`normalize_edge_weights: true`)
- Edge weights reflect co-occurrence frequency and extraction confidence
- Graph pruning parameters allow filtering:
  - `min_edge_weight_pct`: Minimum weight percentile floor
  - `min_node_degree`: Minimum connections required
  - `max_node_degree_std`: Maximum degree standard deviation cap

### Claim Extraction and Verification

- **Disabled by default** (`extract_claims.enabled: false`)
- When enabled, extracts factual claims from text with:
  - Claim description
  - Subject entity
  - Object entity (if applicable)
  - Status: TRUE / FALSE / UNKNOWN / SUSPECTED
  - Time bounds (start/end dates if applicable)
- Useful for fact-checking, temporal analysis, and investigative research
- Adds additional LLM cost per text unit

---

## 1.2 Query Modes Detailed

### Global Search

**Purpose**: Answering holistic, corpus-wide questions that require reasoning across the entire dataset.

**How it works (Map-Reduce)**:
1. **Map Phase**: The system retrieves community summaries at a predetermined hierarchy level. Each community report is sent to the LLM with the user query, producing a partial answer.
2. **Reduce Phase**: All partial answers are combined and synthesized by the LLM into a final coherent response.

**Scaling behavior:**
- Cost scales linearly with the number of communities at the selected level
- Higher hierarchy levels = fewer communities = lower cost but less detail
- Lower hierarchy levels = more communities = higher cost but finer detail

**Token consumption:**
- Each community report consumes tokens in the map phase
- The reduce phase consumes tokens proportional to the number of map outputs
- Typical: 50K-200K+ tokens per global query depending on corpus size and level

**Dynamic Community Selection** (improvement introduced in 2025):
- Instead of processing all communities, uses relevance scoring to select only pertinent communities
- Reduces cost significantly while maintaining quality
- Configurable via `dynamic_search_threshold`, `dynamic_search_keep_parent`, `dynamic_search_num_repeats`

**Best for**: "What are the major themes across all documents?", "What trends have emerged?", "Summarize the entire corpus around topic X"

### Local Search

**Purpose**: Answering specific questions about particular entities and their neighborhoods.

**How it works:**
1. **Entity Identification**: The query is embedded and matched against entity descriptions via vector similarity
2. **Neighborhood Expansion**: For the top-k matched entities, the system retrieves:
   - Connected entities (neighbors in the graph)
   - Relationships between them
   - Associated text units (source chunks)
   - Relevant community reports
   - Entity covariates (if claim extraction was enabled)
3. **Context Assembly**: All retrieved information is assembled into a context window with configurable proportions:
   - `text_unit_prop`: Weight for raw text chunks
   - `community_prop`: Weight for community summaries
4. **LLM Generation**: The assembled context plus query is sent to the LLM

**Performance**: Faster and cheaper than Global Search since it focuses on a graph neighborhood rather than the entire graph.

**Best for**: "What is the relationship between Entity A and Entity B?", "Tell me about Entity X and its connections"

**Key parameters:**
| Parameter | Description |
|-----------|-------------|
| `top_k_entities` | Number of seed entities to retrieve |
| `top_k_relationships` | Number of relationships to include |
| `max_context_tokens` | Maximum context window size |
| `text_unit_prop` | Proportion of context allocated to text units |
| `community_prop` | Proportion allocated to community reports |
| `conversation_history_max_turns` | Chat history to include |

### DRIFT Search (Dynamic Reasoning and Inference with Flexible Traversal)

**Purpose**: Combines global breadth with local depth for queries requiring both.

**Three-phase process:**

**Phase A -- Primer:**
- Query is compared against top-K semantically relevant community reports
- LLM generates an initial answer AND follow-up questions for deeper exploration
- This provides a global context "starting point"

**Phase B -- Follow-Up:**
- Each follow-up question triggers a local search
- Produces intermediate answers and more targeted follow-up questions
- Iterates to configurable depth (`n_depth`)
- Each iteration increases specificity
- Confidence metrics guide query expansion decisions

**Phase C -- Output Hierarchy:**
- All intermediate answers are ranked and organized hierarchically
- Final synthesis balances global insights with local specifics
- Results reflect both breadth and depth

**When DRIFT outperforms Local Search:**
- Queries requiring both entity-specific detail AND broader context
- Questions where the answer spans multiple communities
- Exploratory queries where the user doesn't know exactly what entity to target

**Key parameters:**
| Parameter | Description |
|-----------|-------------|
| `drift_k_followups` | Number of follow-up results to retrieve |
| `primer_folds` | Number of search priming batches |
| `n_depth` | Number of search iteration depth levels |
| `local_search_text_unit_prop` | Text unit weight in local sub-searches |
| `local_search_community_prop` | Community weight in local sub-searches |
| `local_search_top_k_mapped_entities` | Entity count per local sub-search |
| `concurrency` | Parallel request count |

### Basic Search

**Purpose**: Standard vector similarity search as a fallback.

**How it works:**
- Embeds the query
- Retrieves top-k most similar text units via vector similarity
- Sends retrieved chunks + query to LLM

**Parameters:**
- `k`: Number of text units to retrieve
- `max_context_tokens`: Context window limit

**When to use**: Simple factual queries, when graph structure adds no value, or as a baseline comparison.

---

## 1.3 Configuration Deep Dive

### Complete YAML Configuration Reference

```yaml
# ============================================================
# MODEL CONFIGURATION
# ============================================================
completion_models:
  default:
    model_provider: openai           # openai | azure | anthropic
    model: gpt-4o-mini               # Model identifier
    type: litellm                    # litellm | mock
    api_key: ${GRAPHRAG_API_KEY}
    api_base: null                   # Custom API endpoint
    api_version: null                # API version (Azure)
    auth_method: api_key             # api_key | azure_managed_identity
    retry:
      type: exponential_backoff      # exponential_backoff | immediate
      max_retries: 7
      base_delay: 2.0               # seconds
      jitter: true
      max_delay: null
    rate_limit:
      type: sliding_window
      period_in_seconds: 60
      requests_per_period: null
      tokens_per_period: null

embedding_models:
  default:
    model_provider: openai
    model: text-embedding-3-small
    # Same sub-parameters as completion_models

# ============================================================
# INPUT CONFIGURATION
# ============================================================
input:
  type: text                         # text | csv | json
  encoding: utf-8
  file_pattern: ".*\\.txt$"         # Regex for file matching
  storage:
    type: file                       # file | memory | blob | cosmosdb
    base_dir: input

# ============================================================
# CHUNKING
# ============================================================
chunking:
  type: tokens                       # tokens | sentence
  size: 1200                         # Max chunk size (CRITICAL parameter)
  overlap: 100                       # Token overlap between chunks
  encoding_model: cl100k_base
  prepend_metadata: []               # Fields to prefix to chunks

# ============================================================
# GRAPH EXTRACTION
# ============================================================
extract_graph:
  completion_model_id: default
  prompt: null                       # Custom prompt template file path
  entity_types:                      # Customize per domain
    - PERSON
    - ORGANIZATION
    - LOCATION
    - EVENT
  max_gleanings: 1                   # Additional extraction passes (0 = none)

# ============================================================
# NLP-BASED EXTRACTION (alternative to LLM)
# ============================================================
extract_graph_nlp:
  normalize_edge_weights: true
  concurrent_requests: 4
  text_analyzer:
    extractor_type: regex_english    # regex_english | syntactic_parser | cfg
    model_name: en_core_web_sm       # spaCy model
    max_word_length: 15
    include_named_entities: true

# ============================================================
# GRAPH PRUNING
# ============================================================
prune_graph:
  min_node_freq: 1                   # Minimum occurrence count
  max_node_freq_std: null            # Frequency variance cap
  min_node_degree: 1                 # Minimum connections
  max_node_degree_std: null          # Degree variance cap
  min_edge_weight_pct: 0.0           # Weight percentile floor
  remove_ego_nodes: false
  lcc_only: false                    # Restrict to largest connected component

# ============================================================
# CLUSTERING
# ============================================================
cluster_graph:
  max_cluster_size: 10               # Community size cap for export
  use_lcc: false
  seed: 42                           # For reproducibility

# ============================================================
# DESCRIPTION SUMMARIZATION
# ============================================================
summarize_descriptions:
  completion_model_id: default
  prompt: null
  max_length: 500                    # Output token limit
  max_input_length: 8000             # Input token limit

# ============================================================
# COMMUNITY REPORTS
# ============================================================
community_reports:
  completion_model_id: default
  graph_prompt: null                 # Graph-based template
  text_prompt: null                  # Text-based template
  max_length: 2000                   # Output token limit
  max_input_length: 8000             # Input token limit

# ============================================================
# CLAIM EXTRACTION (off by default)
# ============================================================
extract_claims:
  enabled: false
  completion_model_id: default
  prompt: null
  description: "Any claims or assertions in the text"
  max_gleanings: 1

# ============================================================
# EMBEDDING
# ============================================================
embed_text:
  embedding_model_id: default
  batch_size: 16
  batch_max_tokens: 8191
  names:
    - text_unit_text
    - entity_description
    - community_full_content

# ============================================================
# VECTOR STORE
# ============================================================
vector_store:
  type: lancedb                      # lancedb | azure_ai_search | cosmosdb
  db_uri: output/lancedb

# ============================================================
# OUTPUT & CACHE
# ============================================================
output:
  type: file
  base_dir: output
  encoding: utf-8

cache:
  type: json                         # json | memory | none
  storage:
    type: file
    base_dir: cache

# ============================================================
# QUERY CONFIGURATION
# ============================================================
local_search:
  completion_model_id: default
  embedding_model_id: default
  text_unit_prop: 0.5
  community_prop: 0.1
  top_k_entities: 10
  top_k_relationships: 10
  max_context_tokens: 12000
  conversation_history_max_turns: 5

global_search:
  completion_model_id: default
  map_max_length: 1000               # Map response word limit
  reduce_max_length: 2000            # Reduce response word limit
  dynamic_search_threshold: 1        # Relevance rating floor (0-5)
  dynamic_search_keep_parent: false
  dynamic_search_num_repeats: 1
  dynamic_search_use_summary: false
  dynamic_search_max_level: -1       # -1 = all levels

drift_search:
  completion_model_id: default
  embedding_model_id: default
  drift_k_followups: 10
  primer_folds: 3
  n_depth: 3
  concurrency: 4

basic_search:
  completion_model_id: default
  embedding_model_id: default
  k: 20
  max_context_tokens: 12000

# ============================================================
# SNAPSHOTS (for debugging/visualization)
# ============================================================
snapshots:
  embeddings: false                  # Export to Parquet
  graphml: true                      # Export GraphML (for Gephi etc.)
  raw_graph: false                   # Pre-merge graph export
```

### Indexing Cost Analysis

**Cost Formula:**
```
Cost = Total_Tokens_Processed x Cost_Per_Token (by model)
```

**Token multiplier rule of thumb:** For a corpus of N tokens, expect **5-10x N** in total LLM token consumption during indexing (entity extraction, summarization, community reports).

**Practical Cost Examples:**

| Corpus Size | Model | Estimated Indexing Cost |
|-------------|-------|----------------------|
| 30,000 words (~38K tokens) | GPT-4o-mini | ~$0.34 |
| 55,000 words (Wizard of Oz) | GPT-4-Turbo | ~$3.29 |
| 55,000 words (Wizard of Oz) | GPT-4o | ~$1.64 |
| 55,000 words (Wizard of Oz) | GPT-4o-mini | ~$0.06 |
| 1M words (~1.2M tokens) | GPT-4o-mini | ~$8.20 |
| 1M+ documents | GPT-4o | $1,000s-$10,000s |

**Per-word cost reference (GPT-4o-mini):** ~$0.0000113/word or ~$0.0000088/token

**Cost estimation before running:**
```bash
graphrag index --estimate-cost
```
This CLI flag previews estimated token counts and API costs before committing to the full pipeline.

**Cost Optimization Strategies:**
1. Use GPT-4o-mini instead of GPT-4-Turbo (50x reduction)
2. Reduce `max_gleanings` from 1 to 0 (fewer extraction passes)
3. Reduce chunk size from 1,200 to 600 tokens
4. Enable caching (`cache.type: json`) to avoid redundant API calls on re-runs
5. Use local inference servers (Ollama, vLLM) via OpenAI-compatible endpoints
6. Use NLP-based extraction (`extract_graph_nlp`) instead of LLM-based for initial passes
7. Consider LazyGraphRAG for 0.1% of full GraphRAG indexing cost

### Graph Visualization and Exploration Tools

- **GraphML export** (`snapshots.graphml: true`): Opens in Gephi, yEd, Cytoscape
- **Parquet export** (`snapshots.embeddings: true`): Analyzable in pandas, DuckDB
- **Neo4j integration**: Import the graph for interactive exploration
- **Azure Cosmos DB**: Gremlin API for graph traversal

---

## 1.4 Performance & Benchmarks

### Comprehensiveness and Diversity Scores

**Microsoft's VIINA Dataset Evaluation:**
| Metric | GraphRAG | Baseline RAG |
|--------|----------|-------------|
| Comprehensiveness | 72-83% | Unable to answer |
| Diversity | 62-82% | Narrow responses |

**Evaluation Methodology**: LLM-as-a-Judge, presenting answer pairs in counterbalanced order, producing win rates.

### Accuracy Benchmarks

| Benchmark | GraphRAG Accuracy | Vector RAG Accuracy | Improvement |
|-----------|------------------|--------------------|----|
| Enterprise KG Accuracy (FalkorDB) | 3.4x baseline | Baseline | 3.4x |
| Schema-heavy queries | High | Near-zero | Infinite gain |
| Multi-hop reasoning | 86% | 32% | +54 pts |
| Single-hop factual | Comparable | Comparable | Minimal |

**GraphRAG-Bench (ICLR 2026):**
- M2hC LF method: ~50-52% average win rates across datasets
- Semiconductor domain: up to 64% wins for diversity, 52% for comprehensiveness
- RAG outperforms GraphRAG on single-hop questions and fine-grained detail retrieval
- GraphRAG is more effective for multi-hop and reasoning-intensive questions

### Query Latency by Mode

| Mode | Typical Latency | Token Cost per Query |
|------|----------------|---------------------|
| Basic Search | 2-5 seconds | Low (~1K-5K tokens) |
| Local Search | 5-15 seconds | Medium (~5K-20K tokens) |
| DRIFT Search | 15-45 seconds | High (~20K-80K tokens) |
| Global Search | 20-120+ seconds | Very High (~50K-200K+ tokens) |

### Indexing Time at Different Corpus Sizes

| Corpus Size | Approximate Indexing Time | Notes |
|-------------|--------------------------|-------|
| 30K words | Minutes | Quick iteration |
| 200K tokens (book) | ~4 hours | Practical limit for testing |
| 1M tokens | Hours to half-day | Production small corpus |
| 10M+ tokens | Days | Requires parallelization |

---

## 1.5 Implementation

### Microsoft graphrag Python Package

```bash
pip install graphrag

# Initialize project
graphrag init --root ./my-project

# Estimate cost before indexing
graphrag index --root ./my-project --estimate-cost

# Run indexing
graphrag index --root ./my-project

# Query
graphrag query --root ./my-project --method local --query "Your question"
graphrag query --root ./my-project --method global --query "Your question"
graphrag query --root ./my-project --method drift --query "Your question"
graphrag query --root ./my-project --method basic --query "Your question"
```

### LlamaIndex PropertyGraphIndex Integration

LlamaIndex provides a native GraphRAG implementation using `PropertyGraphIndex`:

```python
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Setup graph store
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687"
)

# Build index with GraphRAG extractors
index = PropertyGraphIndex.from_documents(
    documents,
    graph_store=graph_store,
    kg_extractors=[GraphRAGExtractor(llm=llm)],
    show_progress=True
)

# Query with GraphRAG engine
query_engine = GraphRAGQueryEngine(
    graph_store=graph_store,
    llm=llm,
    index=index
)
response = query_engine.query("Your question")
```

**Key LlamaIndex components:**
- `GraphRAGExtractor`: Extracts triples (subject-relation-object) and enriches with descriptions
- `GraphRAGStore` (extends `Neo4jPropertyGraphStore`): Applies Leiden community detection + LLM summarization
- `GraphRAGQueryEngine`: Routes queries to community summaries, synthesizes across communities

### LangChain Integration

```python
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

# Build knowledge graph
llm = ChatOpenAI(model="gpt-4o")
transformer = LLMGraphTransformer(llm=llm)
graph_documents = transformer.convert_to_graph_documents(documents)

graph = Neo4jGraph()
graph.add_graph_documents(graph_documents, include_source=True)

# Query with hybrid retrieval
from langchain_community.vectorstores import Neo4jVector
vector_store = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    node_label="Entity",
    text_node_properties=["description"],
    embedding_node_property="embedding"
)
```

### Neo4j Integration Patterns

**Production architecture requires three synchronized indexes:**
1. **Text Index**: Raw document chunks for full-text search
2. **Vector Index**: Embeddings for semantic similarity (1536-dimensional)
3. **Graph Index**: Entities and relationships for structural traversal

**Neo4j GraphRAG Python package:**
```bash
pip install neo4j-graphrag
```
First-party library from Neo4j with pipeline components for entity extraction, embedding generation, and graph creation.

**Deployment options:**
- Neo4j Aura (managed cloud) + Azure AI Foundry
- Self-hosted Neo4j + Docker containers
- Neo4j Community Edition for development

### Incremental Updates (Current State)

**Status as of early 2026:**
- Full incremental indexing is in development; a `graphrag.append` command is planned
- Current workaround: Re-run full indexing (expensive for large corpora)
- **FastGraphRAG** supports incremental updates by upserting new files into the existing graph
- **LightRAG** supports real-time incremental updates natively
- The planned `append` command will minimize community recomputes, with worst-case degrading to full re-index

**Best practice for now:** Use caching (`cache.type: json`) to skip already-processed text units on re-runs.

---

## 1.6 Limitations and Edge Cases

### Practical Corpus Size Limits

| Corpus Size | Feasibility | Notes |
|-------------|-------------|-------|
| < 100K tokens | Easy | Quick iteration, low cost |
| 100K - 1M tokens | Practical | Standard production use |
| 1M - 10M tokens | Challenging | High cost, multi-hour indexing |
| 10M+ tokens | Enterprise only | Requires significant budget and infrastructure |

### Cost at Scale (1M+ Documents)

- A 1M-word corpus costs ~$8.20 with GPT-4o-mini for indexing alone
- At 1M documents (assuming ~500 words each = 500M words), estimated cost: **$5,000-$50,000+** depending on model
- Global queries on massive corpora can consume 200K+ tokens per query
- **Mitigation**: Use LazyGraphRAG (0.1% indexing cost) or local models

### Entity Resolution Challenges

This is the **single biggest quality risk** in GraphRAG:

- Entities matched primarily by name -- "Dr. Smith" appearing 847 times may be one person or many
- **Critical accuracy threshold**: Below ~85% entity resolution accuracy, the knowledge graph becomes "toxic"
- **Error compounding is exponential**, not linear:
  - At 95% ER accuracy: 5-hop queries achieve 77% accuracy
  - At 85% ER accuracy: 5-hop queries drop to 44% accuracy
  - A single misidentified entity poisons every traversal path through it

**Mitigation strategies:**
1. Human-in-the-loop validation for critical entity types
2. Monitor entity merge rates and resolution accuracy
3. Intent-based query routing (not all queries need graph traversal)
4. Start with vector RAG, add GraphRAG only when relational queries are demonstrated

### Temporal Knowledge Graph Challenges

- GraphRAG does not natively handle temporal evolution of entities
- An entity's properties may change over time (CEO changes, company merges)
- Claim extraction with time bounds partially addresses this, but is limited
- No built-in mechanism for "as of date X" queries

### When GraphRAG Gives Worse Results Than Standard RAG

1. **Simple factual lookups**: When the answer is in a single passage, vector RAG is faster and often more accurate
2. **Fine-grained detail retrieval**: RAG outperforms GraphRAG on granular, specific questions
3. **Low entity density text**: Narrative or conversational text without clear entities yields sparse, unhelpful graphs
4. **Small corpora (< 10 documents)**: Overhead of graph construction provides no benefit
5. **Rapidly changing data**: Indexing latency means stale graphs
6. **Single-hop questions**: No advantage from graph traversal

---

# Part 2: RAPTOR

**Paper**: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)
**Repository**: [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)
**Authors**: Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning (Stanford)

## 2.1 Algorithm Deep Dive

### Recursive Clustering Process (Step by Step)

1. **Leaf Node Creation**: Corpus is divided into chunks of **100 tokens** each (with sentence boundary preservation)
2. **Embedding**: Each chunk is embedded using **SBERT (multi-qa-mpnet-base-cos-v1)**
3. **Dimensionality Reduction**: UMAP reduces embedding dimensions for clustering
4. **GMM Clustering**: Gaussian Mixture Models cluster the embeddings
5. **Summarization**: Each cluster is summarized by an LLM (GPT-3.5-turbo)
6. **New Node Creation**: Summaries become new parent nodes in the tree
7. **Recursion**: Steps 2-6 repeat on the parent nodes until no further clusters can be formed

**Average cluster size**: ~6.7 nodes per parent
**Compression ratio**: 0.28 summary-to-child ratio (72% compression per level)
**Typical tree depth**: 3-5 levels depending on corpus size

### Gaussian Mixture Model (GMM) Clustering Details

**Why GMM over K-means:**
- GMM supports **soft clustering**: a node can belong to multiple clusters (probabilistic assignment)
- This is critical because a text chunk may be relevant to multiple themes
- K-means forces hard assignment, losing multi-topic relationships

**GMM Parameter Selection:**
1. **Bayesian Information Criterion (BIC)** determines optimal number of clusters
2. **Expectation-Maximization (EM)** estimates GMM parameters:
   - Means (cluster centers)
   - Covariances (cluster shapes)
   - Mixture weights (relative cluster sizes)
3. The number of clusters is not preset -- BIC selects it automatically per level

**Soft vs Hard Clustering Implications:**
- **Soft clustering**: A text chunk about "AI regulation in the EU" can belong to both an "AI technology" cluster and a "European policy" cluster
- This means the same leaf node may be summarized into multiple parent nodes
- Results in a DAG (directed acyclic graph) rather than a strict tree, though the paper describes it as a tree
- Improves recall at the cost of some redundancy

### Summarization at Each Level

- **Model used**: GPT-3.5-turbo for summarization (can be replaced)
- Each cluster's constituent text chunks are concatenated and sent to the LLM with a summarization prompt
- The prompt asks for a concise summary that captures the key information across all chunks in the cluster
- Higher-level summaries become increasingly abstract
- The summary token count scales linearly with document length

### Tree Structure Details

| Level | Content | Granularity |
|-------|---------|-------------|
| Level 0 (Leaves) | Original 100-token chunks | Most specific |
| Level 1 | Summaries of ~6-7 leaf clusters | Moderate detail |
| Level 2 | Summaries of Level 1 clusters | High-level themes |
| Level 3+ | Meta-summaries | Most abstract |

- NarrativeQA experiments show nodes retrieved from layers 0-4
- **57.36% of retrieved nodes are non-leaf nodes** (using DPR), confirming the value of the hierarchy

---

## 2.2 Configuration Parameters

### Leaf Node Chunk Size Optimization

| Chunk Size | Trade-off |
|------------|-----------|
| 50 tokens | Very granular, many nodes, deeper tree, higher cost |
| 100 tokens | Paper default -- good balance of granularity and efficiency |
| 200 tokens | Fewer nodes, shallower tree, lower cost, less granular |
| 500 tokens | Coarse -- may miss fine-grained information |

The paper uses **100 tokens** with sentence boundary preservation.

### Clustering Algorithm Selection

| Algorithm | Pros | Cons |
|-----------|------|------|
| GMM (paper default) | Soft clustering, automatic K selection via BIC | More computationally expensive |
| K-means | Faster, simpler | Hard clustering, requires K preset |
| HDBSCAN | Density-based, handles noise | May produce many unclustered outliers |

### Key Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| Chunk size | Leaf node token count | 100 tokens |
| Embedding model | SBERT variant | multi-qa-mpnet-base-cos-v1 |
| Summarization model | LLM for summaries | GPT-3.5-turbo (or GPT-4 for quality) |
| Clustering algorithm | GMM by default | GMM with BIC |
| UMAP dimensions | Reduced dims for clustering | Default UMAP settings |
| Max context tokens | Retrieval budget | 2000 tokens |
| Retrieval strategy | Tree traversal vs collapsed | Collapsed (better performance) |
| Top-k (traversal) | Nodes per level in traversal | 1-3 per level |

### Tree Depth Limits

- No explicit depth limit in the algorithm -- recursion stops when no further clusters can be formed
- Practically, depth is 3-5 levels for typical corpora
- Very large corpora (books, multi-document) may produce 5+ levels
- Each additional level adds cost (LLM summarization calls) and build time

### Retrieval Strategies

**Tree Traversal:**
1. Start at root nodes
2. Select top-k most similar nodes at each level
3. Descend to their children
4. Repeat until reaching leaves or token budget
- Provides control over breadth (k) and depth (d)
- May miss relevant information in unvisited branches

**Collapsed Tree (recommended):**
1. Flatten all nodes from all levels into a single set
2. Compute cosine similarity between query and ALL nodes
3. Select nodes greedily until token budget (2000 tokens) is reached
- Consistently outperforms tree traversal in benchmarks
- Retrieves information at the correct granularity level automatically
- Drawback: Requires similarity search over all nodes (mitigated by FAISS)

---

## 2.3 Benchmarks

### QuALITY Benchmark (Multiple-Choice QA on Medium-Length Passages)

| Method | Accuracy (Test) | Accuracy (Hard Subset) |
|--------|-----------------|----------------------|
| **RAPTOR + GPT-4** | **82.6%** | **76.2%** |
| Previous SOTA (CoLISA) | 62.3% | -- |
| Improvement | **+20.3% absolute** | -- |

### QASPER Benchmark (Full-Text NLP Papers)

| Method | F-1 Match |
|--------|-----------|
| **RAPTOR + GPT-4** | **55.7%** |
| RAPTOR + GPT-3 | 53.1% |
| RAPTOR + UnifiedQA | 36.6% |
| Previous SOTA (CoLT5 XL) | 53.9% |
| DPR baseline | 53.0% |
| BM25 baseline | 50.2% |

RAPTOR margins over baselines:
- vs DPR: +1.8 to +4.5 points across models
- vs BM25: +5.5 to +10.2 points across models

### NarrativeQA Benchmark (Free-Text QA on Books/Movies)

| Method | ROUGE-L | BLEU-1 | BLEU-4 | METEOR |
|--------|---------|--------|--------|--------|
| RAPTOR + SBERT | 30.87% | -- | -- | -- |
| SBERT alone | 29.26% | -- | -- | -- |
| BM25 + RAPTOR | 27.93% | -- | -- | -- |
| DPR + RAPTOR | **30.94%** | -- | -- | -- |
| RAPTOR + UnifiedQA | 30.8% | 23.5% | 6.4% | **19.1% (SOTA)** |

### Comparison with GraphRAG on Overlapping Tasks

| Dimension | RAPTOR | GraphRAG |
|-----------|--------|----------|
| Multi-hop reasoning | Good (hierarchical context) | Excellent (graph traversal) |
| Corpus-wide summarization | Limited (tree is document-level) | Excellent (community reports) |
| Retrieval speed | Fastest among advanced methods | Slower (graph + LLM overhead) |
| Indexing cost | Moderate (LLM summarization) | High (entity extraction + community reports) |
| Accuracy on complex queries | High (QuALITY +20%) | Higher on multi-hop (86% vs 32% baseline) |
| Single-document deep QA | Excellent | Overkill |

---

## 2.4 Implementation

### Official raptor-rag Package

```bash
pip install raptor-rag
```

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel

# Configure
config = RetrievalAugmentationConfig(
    summarization_model=YourSummarizationModel(),
    qa_model=YourQAModel(),
    embedding_model=YourEmbeddingModel(),
    tree_builder_type="cluster",  # or "balanced"
)

# Build tree
ra = RetrievalAugmentation(config=config)
ra.add_documents(text)

# Query
answer = ra.answer_question("Your question")
```

### LlamaIndex RAPTOR Integration (RaptorRetriever)

```bash
pip install llama-index-packs-raptor
```

```python
from llama_index.packs.raptor import RaptorRetriever

retriever = RaptorRetriever(
    documents=documents,
    embed_model="text-embedding-3-small",
    llm=OpenAI(model="gpt-3.5-turbo"),
    vector_store=vector_store,           # Optional persistent store
    similarity_top_k=5,
    mode="collapsed",                    # "collapsed" or "tree_traversal"
)

nodes = retriever.retrieve("Your question")
```

### RAGFlow RAPTOR Support

RAGFlow provides built-in RAPTOR support:
- Enable via configuration toggle
- Handles chunking, clustering, and summarization automatically
- Integrates with RAGFlow's existing retrieval pipeline

### Cost Analysis: LLM Calls for Tree Construction

| Operation | LLM Calls | Tokens Per Call |
|-----------|-----------|-----------------|
| Leaf embedding | 0 (SBERT, local) | N/A |
| Level 1 summarization | ~N/6.7 (where N = leaf count) | ~700-1500 per cluster |
| Level 2 summarization | ~N/45 | ~700-1500 per cluster |
| Level 3+ summarization | Diminishing | ~700-1500 per cluster |

**Total LLM calls scale as approximately**: `N * (1 + 1/6.7 + 1/45 + ...)` which converges quickly.

**Cost estimate for a 100-page document** (~50K tokens):
- ~500 leaf nodes (at 100 tokens each)
- ~75 Level 1 summaries
- ~11 Level 2 summaries
- ~2 Level 3 summaries
- Total: ~88 LLM calls for tree construction
- At GPT-3.5-turbo pricing: **< $0.50**

Token expenditure and build time scale **linearly** with document length.

---

## 2.5 Limitations

### Indexing Time and Cost

- Tree construction for a 200K-token book takes **> 1 hour**
- Impractical for real-time interactions or frequently updated content
- Each level of the tree requires a full round of LLM summarization calls
- Cost is moderate but non-trivial for large corpora

### Tree Staleness When Documents Update

- **No incremental update mechanism**: Changing a single document requires rebuilding affected portions of the tree
- Summaries at higher levels become stale when underlying documents change
- No built-in versioning or diff-based update
- **Workaround**: Rebuild the tree periodically (batch updates)

### When Flat Retrieval Outperforms Tree Retrieval

1. **Simple factual questions**: When the answer is in a single chunk, the tree adds no value
2. **Very short documents**: Tree has minimal depth, collapsed tree is essentially flat retrieval
3. **Highly specific technical queries**: The summarization process may lose precise technical details
4. **Queries about rare/specific entities**: Tree summaries may not preserve entity-level detail

### Memory Requirements for Large Trees

- All nodes (all levels) must be embedded and stored
- For a 1M-token corpus:
  - ~10,000 leaf nodes
  - ~1,500 Level 1 nodes
  - ~225 Level 2 nodes
  - ~34 Level 3 nodes
  - Total: ~11,759 nodes, each with a 768-dim embedding
  - Memory: ~36 MB for embeddings alone (manageable)
- FAISS or similar library needed for efficient nearest-neighbor search at scale

### Flat Tree Structure Problem

- RAPTOR tends to produce relatively **flat tree structures** for some document types
- May not adequately represent complexity in large, multi-topic documents
- Mitigation: Adjust chunk size and clustering parameters

---

# Part 3: Other Graph-Based and Hierarchical Approaches

## LightRAG

**Repository**: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
**Released**: October 2024, Hong Kong University

**Architecture:**
- Constructs a knowledge graph via LLM-based entity/relation extraction (similar to GraphRAG)
- Uses **dual-level retrieval**: low-level (specific entities) and high-level (abstract themes)
- Vector store supports keyword matching alongside semantic search
- **Key differentiator**: Streamlined incremental updates by unioning new documents into existing graph

**Performance vs GraphRAG:**
| Metric | LightRAG | GraphRAG |
|--------|----------|----------|
| Query latency | ~80ms (~30% faster than standard RAG) | Seconds to minutes |
| Incremental update time | ~50% less than full rebuild | Full rebuild required |
| Indexing cost | Much lower (simpler extraction) | 5-10x corpus tokens |
| Multi-hop reasoning | Good but may miss complex chains | Excellent |
| Community detection | None (relies on graph structure) | Leiden hierarchy |

**When to use**: Budget-constrained, frequently updated data, need for real-time queries.

## LazyGraphRAG (Microsoft)

**Architecture:**
- Defers ALL LLM usage to query time
- Indexing uses only NLP noun-phrase extraction (no LLM calls)
- At query time: iterative deepening with LLM-based relevance assessment
- Combines vector search with graph community structure

**Cost comparison:**
| Metric | LazyGraphRAG | Full GraphRAG |
|--------|-------------|---------------|
| Indexing cost | 0.1% of GraphRAG | Baseline |
| Query cost (Z500 config) | 4% of GraphRAG Global | Baseline |
| Answer quality | Outperforms at Z500 | Baseline |

**When to use**: One-off queries, streaming data, budget-constrained, exploratory analysis.

## FastGraphRAG

- 27x faster than standard GraphRAG
- 40% more accurate retrieval
- Uses PageRank-style algorithms for relevance scoring
- NLP-based entity extraction (no LLM for extraction)
- Supports incremental updates natively

## KG-RAG (Knowledge Graph RAG)

- Uses pre-existing knowledge graphs (e.g., domain ontologies, Wikidata)
- Does not build KG from documents -- leverages existing structured knowledge
- Best when authoritative KGs already exist for the domain
- Lower cost (no KG construction), but limited to knowledge in the existing graph

## LlamaIndex PropertyGraphIndex

- Modular framework for building GraphRAG pipelines
- Supports multiple KG extractors (LLM-based, NLP-based, custom)
- Multiple retrievers (vector, keyword, graph traversal, custom)
- Native Neo4j integration via `Neo4jPropertyGraphStore`
- Leiden community detection built-in
- Most flexible option for custom implementations

## Neo4j GraphRAG Package

```bash
pip install neo4j-graphrag
```
- First-party library from Neo4j
- Pipeline components: entity extraction, embedding, graph creation, retrieval
- Hybrid vector + graph search out of the box
- Long-term support guaranteed by Neo4j
- Best for teams already using Neo4j

## HippoRAG

- Neurobiologically-inspired retrieval
- 10-30x cheaper multi-hop reasoning than GraphRAG
- Mimics hippocampal memory indexing
- Better for conversational, evolving query contexts

## PathRAG

- Flow-based graph pruning
- Cuts context by 44% while maintaining accuracy
- Focuses on shortest-path and most-relevant-path retrieval
- Good for reducing token consumption in graph-heavy queries

## Comparison Table: All Graph/Hierarchical Approaches

| Approach | Indexing Cost | Query Cost | Multi-hop | Incremental Updates | Best For |
|----------|-------------|-----------|-----------|--------------------|----|
| **GraphRAG** (Microsoft) | Very High (5-10x corpus) | High (Global), Medium (Local) | Excellent | Planned, not ready | Corpus-wide summarization, complex reasoning |
| **RAPTOR** | Moderate (LLM summaries) | Low (vector search) | Good | No (full rebuild) | Single-document deep QA, hierarchical context |
| **LightRAG** | Low-Medium | Low (~80ms) | Good | Yes (native) | Budget-constrained, real-time, frequent updates |
| **LazyGraphRAG** | Minimal (0.1% of GraphRAG) | Low-Medium (query-time LLM) | Good | Trivial (re-extract NPs) | Exploratory analysis, streaming data |
| **FastGraphRAG** | Low (NLP-based) | Low | Good | Yes (upsert) | Speed-critical, large corpora |
| **KG-RAG** | None (uses existing KG) | Low | Depends on KG | N/A | Domains with existing ontologies |
| **LlamaIndex PropertyGraph** | Configurable | Configurable | Configurable | Partial | Custom implementations, flexibility |
| **Neo4j GraphRAG** | Medium-High | Medium | Good | Manual | Neo4j-native teams, enterprise |
| **HippoRAG** | Medium | Low | Excellent | Partial | Conversational multi-hop, budget |
| **PathRAG** | Medium | Low (44% less context) | Good | Partial | Token-constrained environments |

---

# Part 4: Decision Matrix

## When to Use What

```
Question: Do you need corpus-wide summarization or cross-document reasoning?
  YES -> GraphRAG (or LazyGraphRAG if budget-constrained)
  NO  -> Continue

Question: Do you need multi-level abstraction within single documents?
  YES -> RAPTOR
  NO  -> Continue

Question: Do you have a pre-existing knowledge graph?
  YES -> KG-RAG or Neo4j GraphRAG
  NO  -> Continue

Question: Do you need real-time updates and low latency?
  YES -> LightRAG or FastGraphRAG
  NO  -> Continue

Question: Is your budget limited?
  YES -> LazyGraphRAG (0.1% indexing cost) or LightRAG
  NO  -> Full GraphRAG for maximum quality

Question: Are your queries primarily simple/factual?
  YES -> Standard vector RAG (no graph needed)
  NO  -> GraphRAG or RAPTOR depending on corpus structure
```

## Cost-Quality Spectrum

```
Cheapest                                                    Most Expensive
|---------|---------|---------|---------|---------|---------|
Vector    LazyGR   LightRAG  RAPTOR   FastGR    GraphRAG  GraphRAG
RAG       (indexing)                              (Local)   (Global)

Lowest Quality                                  Highest Quality
(for complex queries)                     (for complex queries)
```

## Key Takeaway

No single architecture wins across all scenarios. The research consensus from 2024-2026 is clear: **Graph RAG's value scales with query complexity**. Choosing the wrong architecture for your query distribution wastes compute while delivering marginal gains over simpler approaches. Start with vector RAG, measure where it fails, and add graph/hierarchical structure only where needed.

---

## Sources

### GraphRAG
- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [GraphRAG Architecture](https://microsoft.github.io/graphrag/index/architecture/)
- [GraphRAG Default Dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)
- [GraphRAG YAML Configuration](https://microsoft.github.io/graphrag/config/yaml/)
- [GraphRAG Methods](https://microsoft.github.io/graphrag/index/methods/)
- [DRIFT Search Documentation](https://microsoft.github.io/graphrag/query/drift_search/)
- [Local Search Documentation](https://microsoft.github.io/graphrag/query/local_search/)
- [GraphRAG Costs Explained (Microsoft Community Hub)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978)
- [LazyGraphRAG (Microsoft Research Blog)](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [DRIFT Search Introduction (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- [GraphRAG Entity Resolution Challenges](https://www.sowmith.dev/blog/graphrag-entity-disambiguation)
- [GraphRAG-Bench (ICLR 2026)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- [Cutting GraphRAG Token Costs by 90%](https://medium.com/graph-praxis/cutting-graphrag-token-costs-by-90-in-production-5885b3ffaef0)
- [GraphRAG Incremental Indexing Discussion](https://github.com/microsoft/graphrag/discussions/511)
- [Reduce GraphRAG Indexing Costs (FalkorDB)](https://www.falkordb.com/blog/reduce-graphrag-indexing-costs/)
- [GraphRAG Implementation at 12M Nodes](https://particula.tech/blog/graphrag-implementation-enterprise-data-platform)

### RAPTOR
- [RAPTOR Paper (arXiv)](https://arxiv.org/abs/2401.18059)
- [RAPTOR Full Paper (ICLR 2024)](https://arxiv.org/html/2401.18059v1)
- [Official RAPTOR Implementation](https://github.com/parthsarthi03/raptor)
- [Expanding Horizons in RAG: Limits of RAPTOR (Stanford CS224N)](https://web.stanford.edu/class/cs224n/final-reports/256925521.pdf)
- [Mastering RAG with RAPTOR (Educative)](https://www.educative.io/blog/mastering-rag-with-raptor)
- [Improving RAG with RAPTOR (VectorHub)](https://superlinked.com/vectorhub/articles/improve-rag-with-raptor)
- [LlamaIndex RAPTOR Pack](https://docs.llamaindex.ai/en/stable/api_reference/packs/raptor/)
- [RAGFlow RAPTOR Support](https://ragflow.io/docs/enable_raptor)
- [Enhancing RAPTOR with Semantic Chunking (Frontiers)](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1710121/full)

### Other Approaches
- [LightRAG (LearnOpenCV)](https://learnopencv.com/lightrag/)
- [GraphRAG vs LightRAG Comparison (Maarga Systems)](https://www.maargasystems.com/2025/05/12/understanding-graphrag-vs-lightrag-a-comparative-analysis-for-enhanced-knowledge-retrieval/)
- [Neo4j GraphRAG Python Package](https://github.com/neo4j/neo4j-graphrag-python)
- [LlamaIndex PropertyGraphIndex Guide](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/)
- [GraphRAG vs HippoRAG vs PathRAG Comparison](https://medium.com/graph-praxis/graphrag-vs-hipporag-vs-pathrag-vs-og-rag-choosing-the-right-architecture-for-your-knowledge-graph-a4745e8b125f)
- [Leiden Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Leiden_algorithm)
- [Leiden Algorithm Documentation](https://leidenalg.readthedocs.io/en/stable/intro.html)
- [Neo4j Leiden GDS](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
