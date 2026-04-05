# Vector Databases for RAG Systems: Comprehensive Research (2025-2026)

> Research Date: April 2026
> Scope: Architecture, benchmarks, pricing, scaling limits, and selection guidance for 10 vector database solutions

---

## Table of Contents

1. [ChromaDB](#1-chromadb)
2. [Pinecone](#2-pinecone)
3. [Qdrant](#3-qdrant)
4. [Weaviate](#4-weaviate)
5. [Milvus](#5-milvus)
6. [pgvector](#6-pgvector)
7. [FAISS](#7-faiss)
8. [LanceDB](#8-lancedb)
9. [Elasticsearch](#9-elasticsearch)
10. [Redis Vector Search](#10-redis-vector-search)
11. [ANN Benchmarks Comparison](#11-ann-benchmarks-comparison)
12. [Distance Metrics Guide](#12-distance-metrics-guide)
13. [Performance at Scale](#13-performance-at-scale)
14. [Cost Comparison](#14-cost-comparison)
15. [Selection Decision Tree](#15-selection-decision-tree)
16. [Framework Integrations](#16-framework-integrations)

---

## 1. ChromaDB

### Architecture Overview

ChromaDB is an open-source, AI-native vector database implementing a **log-structured architecture**. Writes append to an immutable write-ahead log while reads query optimized segment indexes (HNSW, SPANN). Background compaction asynchronously materializes logs into segments. ChromaDB is primarily a **single-node, embedded database** designed for developer productivity and rapid prototyping.

- **Language**: Python (with Rust components)
- **Index Engine**: Fork of hnswlib
- **Storage**: SQLite (metadata) + HNSW index files
- **Deployment**: Embedded in-process, client/server mode, or Docker

### HNSW Configuration

ChromaDB exposes only three HNSW parameters:

| Parameter | Default | Description |
|---|---|---|
| `M` | 16 | Max connections per node in the graph |
| `efConstruction` | 100 | Search depth during index build |
| `efSearch` | 10 | Search depth during query |

No alternative indexes (IVF, PQ) are available. The HNSW index grows but **never shrinks** -- deleting vectors does not reclaim memory. The only way to reclaim space is to recreate the collection entirely.

### Benchmark Numbers

| Instance Type | RAM | Max Vectors (1024d) | Query Latency (mean/p99.9) | Insert Latency (mean/p99.9) | Cost/Month |
|---|---|---|---|---|---|
| r7i.2xlarge | 64 GB | ~15M | 5ms / 7ms | 112ms / 405ms | ~$387 |
| t3.2xlarge | 32 GB | ~7.5M | 5ms / 33ms | 149ms / 520ms | ~$243 |
| t3.xlarge | 16 GB | ~3.6M | 4ms / 7ms | 159ms / 530ms | ~$122 |
| t3.large | 8 GB | ~1.7M | 4ms / 10ms | 199ms / 633ms | ~$61 |
| t3.medium | 4 GB | ~700K | 5ms / 18ms | 191ms / 722ms | ~$31 |
| t3.small | 2 GB | ~250K | 8ms / 29ms | 231ms / 1280ms | ~$16 |

- **Memory formula**: N = R x 0.245 (N = max vectors in millions at 1024d; R = RAM in GB)
- **QPS**: ~180 QPS in comparative benchmarks
- **Metadata filtering**: 450ms latency for complex filters (vs. 65ms for Qdrant)
- **Optimal batch size**: 50-250 embeddings per insert

### Scaling Limits

- **Max tested**: ~7M embeddings (stable and fast)
- **Practical limit**: <1M vectors for most use cases
- **No native clustering or replication**
- **Memory-bound**: HNSW requires full index in RAM; swapping to disk causes rapid performance degradation
- **SPANN index**: Available for distributed deployments where full index cannot fit in single-node memory

### Metadata Filtering

- Supports `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin` operators
- Supports `$and`, `$or` logical operators
- Filtering is **post-search** (filter after ANN retrieval), which hurts precision on selective filters
- Complex filter performance is poor (~450ms)

### Hybrid Search

- **Not natively supported**. ChromaDB is vector-only search
- No BM25 or keyword search integration

### Pricing

- **Open source**: Free (Apache 2.0 license)
- **Self-hosted cost**: Infrastructure only (see benchmark table above)
- **No managed cloud offering** as of early 2026

### Self-Hosting Requirements

- Minimum 2 GB RAM recommended
- Disk: approximately equal to RAM size plus overhead
- Python 3.8+ runtime
- Docker support available
- No GPU required

### Best For

- Prototyping and development
- Small-scale RAG applications (<1M vectors)
- Quick experimentation with embeddings
- Notebook environments and local development

### Limitations Summary

- Single-node only (no horizontal scaling)
- HNSW index never shrinks on deletion
- Only 3 tunable HNSW parameters
- No hybrid search
- Poor complex metadata filter performance
- No managed cloud service

---

## 2. Pinecone

### Architecture Overview

Pinecone is a **fully managed, cloud-native vector database** that separates storage and compute. Since 2024, Pinecone has focused on its **serverless architecture**, which automatically scales based on usage.

- **Architecture**: Serverless (primary) and Dedicated Read Nodes (DRN)
- **Cloud regions**: AWS (us-east-1, us-west-2, eu-west-1), GCP, Azure
- **Namespace support**: Data partitioning within indexes for multi-tenant applications
- **Max dimensions**: 20,000

### Serverless vs. Dedicated Read Nodes

| Feature | Serverless (On-Demand) | Dedicated Read Nodes (DRN) |
|---|---|---|
| Pricing model | Pay-per-request (RU/WU/storage) | Fixed hourly per node |
| Best for | Variable/bursty workloads | Sustained high-QPS workloads |
| Scaling | Automatic | Manual node provisioning |
| Latency predictability | Variable | Consistent |
| Cost efficiency | Better for low-medium traffic | Better for high sustained traffic |

### Benchmark Numbers

| Configuration | Vectors | QPS | Latency (p50/p99) |
|---|---|---|---|
| p1 pod (768d, 1M vectors) | 1M | ~20 QPS | <120ms p95 |
| p2 pod (256d, 10M vectors) | 10M | ~1,000 QPS | <10ms |
| DRN (135M vectors) | 135M | 600 QPS | 45ms p50 / 96ms p99 |
| DRN (1.4B vectors, filtered) | 1.4B | 5,700 QPS | 26ms p50 / 60ms p99 |
| s1 pod (100M+ vectors) | 100M+ | Varies | <500ms p95 |

### Scaling Limits

- **Max vectors**: Billions (1.4B benchmarked)
- **Max dimensions**: 20,000
- **Max namespaces per index**: 100 (starter), 10,000+ (standard/enterprise)
- **Max indexes**: 5 (starter), 100+ (paid)

### Filtering Capabilities

- Metadata filtering with equality, range, set membership operators
- Filtering is **integrated into the ANN search** (not post-filter)
- Supports up to 40 metadata fields per vector
- Filter selectivity affects RU cost and query performance

### Hybrid Search

- **Supported** via sparse-dense vectors
- Combines dense embeddings with sparse (BM25/SPLADE) representations
- Additional RU cost scales with namespace size and non-zero sparse dimensions
- Alpha parameter controls dense vs. sparse weighting

### Pricing Model

| Plan | Minimum | Read Units | Write Units | Storage |
|---|---|---|---|---|
| Starter (Free) | $0 | 1M RU/mo free | 2M WU/mo free | 2 GB free |
| Standard | $50/mo | $16/M RUs | $4/M WUs | $0.33/GB/mo |
| Enterprise | $500/mo | $24/M RUs | $6/M WUs | $0.33/GB/mo |

- Query cost scales linearly with namespace size: 1 RU per 1 GB of namespace (minimum 0.25 RU/query)
- Hybrid search incurs additional RUs

### Self-Hosting

- **Not available**. Pinecone is managed-only

### Best For

- Production RAG applications needing managed infrastructure
- Teams without vector database operational expertise
- Applications requiring hybrid search out-of-the-box
- Variable-traffic workloads (serverless)
- Large-scale deployments with DRN

---

## 3. Qdrant

### Architecture Overview

Qdrant is a **high-performance vector database written entirely in Rust**. It uses a custom storage engine (Gridstore) with SIMD optimizations for maximum throughput. Qdrant supports both self-hosted and managed cloud deployments.

- **Language**: Rust
- **APIs**: REST, gRPC, official clients (Python, JS, Go, Rust, Java, C#)
- **Storage engine**: Gridstore (custom)
- **Deployment**: Single-node, distributed cluster, Docker, Kubernetes, Qdrant Cloud

### HNSW + Quantization

**HNSW enhancements (2025):**
- GPU-accelerated HNSW indexing (up to 10x faster ingestion)
- Incremental HNSW indexing for upsert-heavy workloads
- HNSW graph compression to reduce memory footprint

**Quantization options:**
- Scalar quantization (INT8): ~4x memory reduction
- Binary quantization: up to 32x compression
- Product quantization: up to 64x compression
- 1.5-bit, 2-bit, and asymmetric quantization (new in 2025)
- Memory reduction: up to 97% with maintained search quality

### Benchmark Numbers

| Metric | Value |
|---|---|
| Peak QPS (tuned, scalar quantization) | ~12,000 QPS |
| QPS (standard config) | ~8,500 QPS |
| Recall | ~98.5% (leading benchmark) |
| p50 latency | ~30.75ms |
| p95 latency | ~36.73ms |
| p99 latency | ~38.71ms |

Qdrant consistently achieves highest RPS and lowest latencies across most benchmark scenarios.

### Scaling Limits

- **Max vectors per node**: 500M+ (with on-disk payload indexing)
- **Max dimensions**: 65,535
- **Sharding**: Automatic and manual shard distribution
- **Replication**: Configurable replication factor per collection
- **Zero-downtime rolling updates**

### Payload Filtering

- Rich payload filtering with nested JSON support
- **Pre-filtering integrated into HNSW traversal** (not post-filter)
- Supports: match, range, geo-bounding box, full-text search (basic)
- Payload indexes for accelerated filtering
- Filter performance: ~65ms for complex metadata queries

### Hybrid Search

- Hybrid fusion search combining dense and sparse vectors
- Multi-vector retrieval support
- Reciprocal Rank Fusion (RRF) for result merging
- Named vectors (multiple vector types per point)

### Pricing Model

| Tier | Cost | Resources | SLA |
|---|---|---|---|
| Free | $0 forever | 0.5 vCPU, 1 GB RAM, 4 GB disk | - |
| Standard | From $25/mo | Flexible scaling, HA, backups | 99.5% |
| Premium | Custom (contact sales) | SSO, Private VPC, advanced security | 99.9% |
| Hybrid Cloud | Custom | Your infrastructure, Qdrant management | Custom |

### Self-Hosting Requirements

- Docker or Kubernetes recommended
- Minimum: 1 vCPU, 1 GB RAM for small datasets
- Production: 4+ vCPU, 16+ GB RAM recommended
- Open source (Apache 2.0)
- VPS costs as low as $5-20/mo for small datasets

### Best For

- High-performance production applications
- Applications requiring advanced filtering
- Self-hosted deployments (Rust efficiency = lower infrastructure costs)
- Multi-vector and hybrid search use cases
- Teams comfortable with operational management

---

## 4. Weaviate

### Architecture Overview

Weaviate is an **open-source, cloud-native vector database** written in Go. It is distinguished by its modular architecture and native GraphQL API. Weaviate was designed from the ground up for hybrid search and AI-native workflows.

- **Language**: Go
- **APIs**: REST, GraphQL, gRPC
- **Module system**: Pluggable vectorizers, rerankers, generators
- **Storage**: Custom LSM-tree based engine

### Module System

Weaviate's modular design allows plugging in pre-built or custom modules:

| Module Category | Examples |
|---|---|
| Vectorizers | text2vec-openai, text2vec-cohere, text2vec-huggingface, text2vec-transformers, img2vec-neural |
| Rerankers | reranker-cohere, reranker-transformers |
| Generators | generative-openai, generative-cohere, generative-palm |
| Other | qna-transformers, ref2vec-centroid |

Modules handle automatic embedding generation, enabling import of raw text/images without pre-computing vectors.

### GraphQL API

- First and most mature GraphQL-native vector database
- Supports Get, Aggregate, and Explore queries
- Complex nested queries with filters and search operators
- Batch operations via REST API

### Hybrid Search

- Native hybrid search combining **BM25 keyword search** and **vector search**
- Configurable alpha parameter (0 = pure keyword, 1 = pure vector)
- Fusion algorithms: rankedFusion (default), relativeScoreFusion
- Single-query hybrid search -- no pipeline stitching required

### Multi-Tenancy

- Strong data isolation with per-tenant shards
- Activity-based tenant states: ACTIVE, INACTIVE, OFFLOADED
- RBAC authorization integrated with OIDC providers
- Auto-tenant activation/deactivation for cost optimization
- Supports thousands of tenants per collection

### Benchmark Numbers

| Metric | Value |
|---|---|
| QPS | ~1,500 QPS (comparative benchmark) |
| Recall | ~97.2% |
| Latency | Sub-50ms (typical) |
| Indexing time | 2.8-3.2x slower than Redis |

### Scaling Limits

- **Horizontal scaling**: Sharding and replication across nodes
- **Multi-tenancy**: Thousands of tenants per cluster
- **Replication factor**: Configurable per collection
- **Max dimensions**: 65,535

### Pricing Model

| Plan | Starting Price | Deployment | SLA |
|---|---|---|---|
| Free Trial | $0 (14 days) | Shared sandbox | - |
| Flex | $45/mo | Shared cloud, pay-as-you-go | 99.5% |
| Premium | $400/mo | Shared or dedicated | 99.95% |

**Usage dimensions:**
- Vector dimensions: $0.00975-$0.01668 per million (varies by compression/region)
- Storage: $0.2125-$0.31875 per GiB
- Backup storage: $0.022-$0.0264 per GiB
- Embeddings add-on: $0.025-$0.065 per million tokens

### Self-Hosting Requirements

- Docker or Kubernetes
- Minimum: 2 vCPU, 4 GB RAM
- Production: 8+ vCPU, 32+ GB RAM recommended
- Open source (BSD-3-Clause)

### Best For

- Applications requiring hybrid search (BM25 + vector)
- Multi-tenant SaaS applications
- Teams wanting integrated vectorization (no separate embedding pipeline)
- GraphQL-native API consumers
- Enterprise deployments requiring RBAC and compliance

---

## 5. Milvus

### Architecture Overview

Milvus is a **cloud-native, distributed vector database** designed for billion-scale similarity search. It separates storage and compute, allowing independent horizontal scaling of query nodes (read-heavy) and data nodes (write-heavy).

- **Language**: Go + C++ (core engine)
- **Architecture**: Disaggregated compute/storage with message queue (Pulsar/Kafka)
- **Components**: Proxy, Query Node, Data Node, Index Node, Root Coord, Query Coord, Data Coord, Index Coord
- **Storage**: MinIO/S3 (object storage) + etcd (metadata)
- **Deployment**: Standalone, distributed cluster, Kubernetes (Milvus Operator), Zilliz Cloud

### GPU Acceleration

- **NVIDIA CAGRA**: GPU-based graph indexing algorithm
- GPU-accelerated via NVIDIA cuVS library
- IVF-PQ, IVF-Flat, Flat, and CAGRA index types on GPU
- GPU index build: up to 12.3x faster than CPU HNSW
- GPU search: up to 4.7x lower latency than CPU

### Benchmark Numbers

| Metric | Value |
|---|---|
| QPS (vs FAISS) | 4.5x higher than FAISS |
| QPS (vs Elasticsearch) | 3-7x higher on BEIR dataset |
| Filter latency (JSON Path Index, 100M+) | 1.5ms mean / 10ms p99 (was 140ms/480ms) |
| Recall | ~97.9% |
| Scaling | Linear with CPU cores (8 to 32) and replicas (1 to 8) |

Milvus 2.6 achieves 3-4x higher throughput than Elasticsearch with equivalent recall rates, with specific workloads reaching 7x higher QPS.

### Scaling Limits

- **Max vectors**: Billions (designed for billion-scale)
- **Max dimensions**: 32,768
- **Horizontal scaling**: Independent scaling of query/data/index nodes
- **Multi-datacenter**: Supported in enterprise deployments
- **Partition keys**: For data distribution and isolation

### Filtering Capabilities

- Boolean expressions on scalar fields
- JSON Path Index (Milvus 2.6): 99% latency reduction for JSON field filtering
- Range, term, prefix, and comparison filters
- Array field filtering
- Expression-based filtering integrated into search

### Hybrid Search

- **Supported**: Dense + sparse vector search
- Multi-vector search with weighted combination
- RRF and weighted scoring fusion
- Full-text search via BM25 (Milvus 2.5+)

### Pricing Model (Zilliz Cloud)

| Tier | Price | Capacity |
|---|---|---|
| Free | $0/mo | 5 GB storage, 2.5M vCUs/mo, 5 collections |
| Serverless | $4/M vCUs + $0.30/GB | Auto-scaling |
| Dedicated (Performance) | ~$65/M vectors/mo | 1.5M vectors/CU, 500-1500 QPS, 10ms latency |
| Dedicated (Capacity) | ~$20/M vectors/mo | 5M vectors/CU, 100-300 QPS, 50-100ms latency |
| Dedicated (Tiered Storage) | ~$7/M vectors/mo | 20M vectors/CU, hot/cold data management |
| Business Critical | Custom | HIPAA, CMEK, priority support |

- Open-source Milvus: Free (Apache 2.0)

### Self-Hosting Requirements

- **Standalone**: 8+ vCPU, 16+ GB RAM, SSD
- **Distributed**: Kubernetes cluster with etcd, MinIO/S3, Pulsar/Kafka
- **Dependencies**: etcd (metadata), MinIO or S3 (storage), Pulsar or Kafka (message queue)
- Significantly more complex than single-binary databases

### Best For

- Billion-scale vector search applications
- GPU-accelerated workloads
- Teams needing distributed architecture with independent scaling
- Enterprise deployments requiring compliance (via Zilliz Cloud)
- High-throughput production workloads

---

## 6. pgvector

### Architecture Overview

pgvector is a **PostgreSQL extension** that adds vector similarity search capabilities to existing PostgreSQL databases. It enables storing embeddings alongside relational data in a single database.

- **Type**: PostgreSQL extension (not standalone database)
- **Language**: C
- **Index types**: HNSW, IVFFlat
- **Distance metrics**: L2 (Euclidean), inner product, cosine, L1 (Manhattan), Hamming, Jaccard
- **Version**: 0.8.0 (latest as of 2025)

### IVFFlat vs. HNSW

| Feature | IVFFlat | HNSW |
|---|---|---|
| Build time | Faster | Slower (2-10x) |
| Memory usage | Lower | Higher |
| Query latency | ~2.4ms | ~1.5ms |
| Recall at k=10 | Good (probe-dependent) | Better |
| Scaling behavior | Linear with probes | Logarithmic |
| Training required | Yes (requires data) | No |
| Incremental updates | Partial (new vectors not in clusters) | Full support |

HNSW generally outperforms IVFFlat in speed-recall tradeoff, especially for high-recall scenarios.

### pgvectorscale

pgvectorscale (by Timescale) extends pgvector with:
- StreamingDiskANN index for larger-than-memory datasets
- Statistical Binary Quantization (SBQ) for 97% memory reduction
- Benchmarked at **471 QPS at 99% recall on 50M vectors** (11.4x better than Qdrant's 41 QPS at same recall in that specific benchmark)

### Benchmark Numbers

| Configuration | QPS | Recall | Latency |
|---|---|---|---|
| HNSW (standard) | Varies | 95-99% | ~1.5ms |
| IVFFlat (standard) | Varies | 90-97% | ~2.4ms |
| Sequential scan | Varies | 100% | ~650ms |
| pgvectorscale (50M, 99% recall) | 471 QPS | 99% | Varies |

### Scaling Limits

- **Max dimensions**: ~2,000 (limited by PostgreSQL 8KB page size)
- **Max dimensions for indexing**: 2,000 (HNSW_MAX_DIM and IVFFLAT_MAX_DIM)
- **Max vectors**: Millions (10M+ performance degrades without optimization)
- **No native sharding for vectors** (relies on PostgreSQL partitioning)
- **TOAST limitation**: Vectors exceeding 8KB page use TOAST, but TOAST-stored vectors cannot be indexed

### Filtering Capabilities

- Full SQL WHERE clause support (leverages PostgreSQL's query planner)
- B-tree, GIN, GiST indexes on metadata columns
- Join operations with other tables
- Rich filtering is a major advantage over purpose-built vector databases

### Hybrid Search

- Combine vector similarity with full-text search (tsvector/tsquery)
- Use PostgreSQL's native full-text search alongside vector search
- Requires manual result fusion (no built-in RRF)

### Pricing Model

- **Free**: Open-source extension (PostgreSQL license)
- **Managed options**: Available on Supabase, Neon, AlloyDB, Amazon Aurora, Azure CosmosDB for PostgreSQL, Timescale Cloud
- **Timescale Cloud** (pgvector + pgvectorscale + pgai): Starting ~$30/mo

### Self-Hosting Requirements

- PostgreSQL 12+ (15+ recommended)
- Install via `CREATE EXTENSION vector;`
- No additional infrastructure beyond PostgreSQL
- Shared memory and work_mem tuning recommended for large datasets

### Best For

- Teams already using PostgreSQL
- Applications needing relational + vector data in one database
- Moderate-scale RAG (<10M vectors)
- Reducing operational complexity (one database for everything)
- Strong SQL filtering requirements

### When NOT to Use

- Billion-scale datasets
- Dimensions > 2,000
- GPU acceleration needed
- Dedicated vector workloads requiring maximum throughput

---

## 7. FAISS

### Architecture Overview

FAISS (Facebook AI Similarity Search) is a **library** (not a database) developed by Meta for efficient similarity search and clustering of dense vectors. It provides the underlying index algorithms used by many vector databases.

- **Language**: C++ with Python bindings
- **Type**: Library (no server, no persistence, no metadata)
- **GPU support**: NVIDIA CUDA via cuVS (since v1.10.0)
- **License**: MIT

### Index Types

| Index | Description | Memory | Speed | Recall | GPU Support |
|---|---|---|---|---|---|
| Flat (IndexFlatL2) | Brute-force exact search | Full vectors | Slowest | 100% | Yes |
| IVF-Flat | Inverted file with flat quantizer | Full vectors per cluster | Fast | High | Yes |
| IVF-PQ | Inverted file + Product Quantization | Compressed (M/2 bytes/vec) | Very fast | Good | Yes |
| HNSW | Hierarchical navigable small world graph | Full vectors + graph overhead | Very fast | Very high | No (CPU only) |
| CAGRA | GPU-native graph index | GPU VRAM | Fastest | Very high | GPU only |
| OPQ | Optimized Product Quantization | Highly compressed | Fast | Moderate | Partial |

### GPU Performance (cuVS, FAISS v1.10.0+)

- IVF build: up to 4.7x faster than classic GPU
- IVF search: up to 8.1x lower latency
- CAGRA build: up to 12.3x faster than CPU HNSW
- CAGRA search: up to 4.7x lower latency than CPU HNSW

### Benchmark Numbers

| Configuration | QPS | Recall |
|---|---|---|
| Deep1B (billion-scale, single core) | ~500 QPS | Varies |
| 10M vectors (L2) | ~3,000 QPS at 90% recall | 90% |
| FAISS (L2, 5 neighbors) | 803 QPS | 98.4% |

### Scaling Limits

- **Max vectors**: Limited by available RAM (or GPU VRAM)
- **No max dimension limit** (practical limits from memory)
- **10 GB flat index = 10 GB of physical memory required**
- **Billion-scale**: Requires sharding across multiple processes/machines (manual)
- **No built-in distribution or replication**

### Filtering Capabilities

- **No metadata filtering**. FAISS is pure vector similarity search
- Workaround: Over-fetch results and filter post-retrieval (slow, imprecise)
- No structured data storage

### Hybrid Search

- **Not supported**. Vector similarity only
- No text search integration

### Pricing

- **Free and open source** (MIT license)
- Cost = infrastructure only

### Limitations Summary

- Library, not database (no CRUD, no persistence, no metadata)
- No built-in metadata filtering
- No built-in concurrency (single-threaded; requires manual thread management)
- HNSW does not support vector removal
- GPU indexes require NVIDIA hardware
- No distributed mode (manual sharding required)
- No hybrid search

### Best For

- Research and experimentation
- Custom vector search infrastructure (building your own database)
- Offline batch processing
- GPU-accelerated search with CAGRA
- Maximum control over index parameters
- Embedding in other applications as a library dependency

---

## 8. LanceDB

### Architecture Overview

LanceDB is an **embedded, serverless vector database** built on the Lance columnar format. It runs in-process (like SQLite for vectors) and is designed for multimodal AI workloads.

- **Language**: Rust (core) + Python/TypeScript SDKs
- **Storage format**: Lance (columnar, alternative to Parquet, optimized for random access)
- **Deployment**: Embedded (in-process), LanceDB Cloud (serverless)
- **Multimodal**: Stores vectors, text, images, videos, audio, point clouds alongside embeddings

### Lance Format

- Modern columnar format optimized for high-speed random access
- Zero-copy reads
- Efficient for managing AI datasets (vectors + documents + images)
- Supports versioning and time travel
- Can store actual data alongside embeddings (not just metadata)

### Benchmark Numbers

| Metric | Value |
|---|---|
| Vector search latency | ~25ms (standard), 3-5ms (high recall) |
| Filtered search latency | ~50ms |
| QPS | 2,000+ QPS (AWS instance) |
| Storage I/O | 1.5M IOPS (2026 benchmark) |

LanceDB is faster than Elasticsearch for both vector search and full-text search in QPS benchmarks.

### Scaling Limits

- **Max vectors**: Petabyte-scale (backed by object storage)
- **Billions of vectors**: Supported with disk-based indexing
- **Storage**: S3, GCS, Azure Blob, local disk
- **No server to manage** in embedded mode

### Filtering Capabilities

- SQL-based filtering
- Metadata filtering alongside vector search
- Full-text search support
- Predicate pushdown for efficient filtered queries

### Hybrid Search

- **Supported**: Vector similarity + full-text search + SQL
- Combines multiple retrieval modes in a single query

### Pricing Model

| Tier | Cost |
|---|---|
| OSS (embedded) | Free (Apache 2.0) |
| LanceDB Cloud (serverless) | Pay for storage + compute (public beta) |
| Enterprise | Annual commitment, custom pricing |

- Embedded mode: Zero infrastructure cost beyond compute and storage
- Cloud: Serverless pay-per-use model

### Self-Hosting Requirements

- Zero infrastructure for embedded mode
- Python 3.8+ or Node.js
- Storage: Local disk, S3, GCS, or Azure Blob
- No server process to manage

### Best For

- Embedded/edge AI applications
- Multimodal search (text + images + video)
- Serverless architectures
- Cost-sensitive deployments (no server overhead)
- Applications needing actual data co-located with vectors
- Rapid prototyping with production-grade performance

---

## 9. Elasticsearch

### Architecture Overview

Elasticsearch is a **distributed search and analytics engine** that added vector search capabilities starting with version 8.0. It uses Apache Lucene under the hood and provides HNSW-based approximate nearest neighbor search alongside its mature text search capabilities.

- **Language**: Java (Lucene core)
- **Index engine**: Apache Lucene (with HNSW, SIMD optimizations since 2025)
- **APIs**: REST, various client libraries
- **Deployment**: Self-hosted, Elastic Cloud, AWS OpenSearch

### Vector Search Capabilities

- **kNN search**: Approximate (HNSW) and exact (script_score brute-force)
- **Max dimensions**: 4,096
- **Retriever API** (8.14+): Combines multiple search types in a single query
- **DiskBBQ** (9.2): Better Binary Quantization for disk-efficient vector search (~15ms at 100 MB memory)
- **ACORN-1**: Fast filtered kNN on large datasets
- **GPU-accelerated indexing**: Tech Preview in 9.3 (early 2026)

### Hybrid Search

Elasticsearch is arguably the most mature platform for hybrid search:

- **BM25 + vector search** in a single query
- **RRF (Reciprocal Rank Fusion)** for combining results (GA in 8.16)
- **Linear combination** of results in ES|QL (9.2)
- **Semantic search** via sparse_vector (ELSER model)
- **Multi-stage retrieval** pipelines
- No pipeline stitching required -- shared filtering layer

### Benchmark Numbers

| Metric | Value |
|---|---|
| vs. OpenSearch (query speed) | Up to 5x faster |
| vs. OpenSearch (throughput) | 3.9x higher on average |
| vs. OpenSearch (filtered, 20M docs) | 8x higher throughput |
| DiskBBQ latency | ~15ms at 100 MB memory |
| vs. Milvus 2.6 throughput | 3-7x lower (Milvus wins) |

### Scaling Limits

- **Horizontal scaling**: Native sharding and replication
- **Max dimensions**: 4,096
- **Cluster size**: Thousands of nodes possible
- **Mature operational tooling**: Decades of production use

### Filtering Capabilities

- Full Lucene query DSL for filtering
- Structured, semi-structured, and high-dimensional data in single query
- ACORN-1 for efficient filtered kNN

### Pricing Model

- **Self-hosted**: Free (Elastic License 2.0 / AGPL)
- **Elastic Cloud**: Starting ~$95/mo (standard deployment)
- **Approximate vector search cost**: ~$3.60/hr for optimized setup

### Self-Hosting Requirements

- JVM (Java 17+)
- Minimum: 4 GB RAM per node
- Production: 32+ GB RAM, SSDs recommended
- Significant operational expertise required
- Mature ecosystem of monitoring and management tools

### Best For

- Applications already using Elasticsearch for text search
- Hybrid search (text + vector) as primary requirement
- Enterprise environments with existing Elastic infrastructure
- Applications requiring mature operational tooling
- Log analytics + vector search combined workloads

---

## 10. Redis Vector Search

### Architecture Overview

Redis provides vector search capabilities through its **Search and Query** module (formerly RediSearch). Being an in-memory data store, Redis delivers ultra-low latency vector search.

- **Language**: C
- **Type**: Module for Redis Stack / Redis Cloud
- **Storage**: In-memory (primary) with optional persistence
- **APIs**: Redis commands (FT.SEARCH, FT.CREATE), client libraries

### Index Types

| Index | Description | Use Case |
|---|---|---|
| FLAT | Exact brute-force search | Small datasets, accuracy-critical |
| HNSW | Approximate graph-based search | Production workloads |
| SVS-VAMANA | Memory-efficient graph index | Large-scale, memory-constrained |

**SVS-VAMANA** (2025): 26-37% total memory savings vs. HNSW at high recall levels, with largest gains from Intel LVQ/LeanVec optimizations.

### Benchmark Numbers

| Metric | Value |
|---|---|
| HNSW latency | Sub-millisecond p50 |
| Indexing time vs. Milvus | 2.8x faster |
| Indexing time vs. Weaviate | 3.2x faster |
| 1B vectors (HNSW, tuned) | ~1.3s median, 95% precision |
| Memory usage | Highest among compared databases |

### Scaling Limits

- **Max dimensions**: 32,768
- **Max vectors**: Millions per shard (memory-bound)
- **Clustering**: Redis Cluster for horizontal scaling
- **Replication**: Redis Sentinel or Cluster replication

### Filtering Capabilities

- Tag, numeric, and geo filters
- Combined vector + attribute filtering in single query
- Full-text search alongside vector search
- JSON document support with JSONPath filtering

### Hybrid Search

- **Supported**: Vector search + full-text search (BM25) + attribute filtering
- Combined in single FT.SEARCH query
- Pre-filtering and post-filtering modes

### Pricing Model

- **Redis Stack (self-hosted)**: Free (source-available license, SSPL)
- **Redis Cloud**: Pay-as-you-go
  - Free tier: 30 MB
  - Fixed plans: Starting ~$7/mo
  - Flexible plans: Starting ~$88/mo
  - Enterprise: Custom pricing, up to 99.999% SLA
- **Scaling 1M vectors**: ~$0.85/mo (self-hosted on SkyPilot)

### Self-Hosting Requirements

- Redis Stack 7.2+ or Redis 8.0+
- Memory: Proportional to dataset size (in-memory)
- Standard Redis deployment knowledge
- Docker, Kubernetes, or bare-metal

### Best For

- Real-time applications requiring sub-millisecond latency
- Caching + vector search combined workloads
- Applications already using Redis
- Session-based recommendations
- Small-to-medium vector datasets (memory cost at large scale)

---

## 11. ANN Benchmarks Comparison

### Comparative Performance Summary (2025-2026)

| Database | QPS (typical) | Recall (k=10) | p50 Latency | p99 Latency | Memory Efficiency |
|---|---|---|---|---|---|
| ChromaDB | ~180 | ~95% | 4-5ms | 7-33ms | Low (HNSW in RAM) |
| Pinecone (DRN) | 600-5,700 | ~97% | 26-45ms | 60-96ms | Managed |
| Qdrant | 8,500-12,000 | ~98.5% | ~31ms | ~39ms | High (quantization) |
| Weaviate | ~1,500 | ~97.2% | Sub-50ms | Varies | Moderate |
| Milvus | 3-7x ES | ~97.9% | Varies | 10ms (filtered) | High (tiered storage) |
| pgvectorscale | 471 (99% recall) | 99% | Varies | Varies | Moderate |
| FAISS | 500-3,000 | 90-98% | 2ms (single core) | Varies | Low (all in RAM) |
| LanceDB | 2,000+ | ~95% | 3-25ms | ~50ms | High (disk-based) |
| Elasticsearch | Varies | ~95% | 15ms (DiskBBQ) | Varies | Moderate (DiskBBQ) |
| Redis | Highest | ~95% | Sub-1ms | Varies | Lowest (all in RAM) |

### Key Benchmarking Tools

| Tool | Focus | Maintainer |
|---|---|---|
| ANN-Benchmarks | Algorithm comparison (single-node) | Erik Bernhardsson |
| VectorDBBench | Full database evaluation (concurrent, filtered) | Zilliz |
| Qdrant Benchmarks | Multi-database comparison | Qdrant |

**Important caveat**: ANN-Benchmarks evaluates index algorithms, not production databases. It does not test filtered search, concurrent access, or operational characteristics. VectorDBBench provides more realistic production-like evaluations including resource consumption and stability.

---

## 12. Distance Metrics Guide

### Supported Metrics by Database

| Database | Cosine | Dot Product | Euclidean (L2) | Others |
|---|---|---|---|---|
| ChromaDB | Yes | Yes (IP) | Yes | - |
| Pinecone | Yes | Yes | Yes | - |
| Qdrant | Yes | Yes | Yes | Manhattan |
| Weaviate | Yes | Yes | Yes | Hamming, Manhattan |
| Milvus | Yes (IP) | Yes | Yes | Jaccard, Hamming, Tanimoto |
| pgvector | Yes | Yes (IP) | Yes | L1, Hamming, Jaccard |
| FAISS | Yes | Yes (IP) | Yes | - |
| LanceDB | Yes | Yes | Yes | - |
| Elasticsearch | Yes | Yes | Yes | L1, Max IP |
| Redis | Yes | Yes (IP) | Yes | - |

### When to Use Each Metric

| Metric | Best For | Key Property |
|---|---|---|
| **Cosine Similarity** | Text similarity, semantic search, document comparison | Direction-only (ignores magnitude). Best for NLP embeddings where document length varies |
| **Dot Product (Inner Product)** | Recommendations, LLM-trained embeddings, collaborative filtering | Considers both direction and magnitude. Equivalent to cosine when vectors are normalized. Faster computation |
| **Euclidean Distance (L2)** | Clustering, anomaly detection, spatial data | Absolute distance in space. Sensitive to magnitude. Good for count-based features |

### Selection Rule of Thumb

1. **Use the metric your embedding model was trained with** (check model documentation)
2. If vectors are **normalized**: cosine and dot product are equivalent; prefer dot product (faster)
3. If magnitude matters (user activity levels, counts): use dot product or Euclidean
4. If only direction matters (semantic similarity): use cosine
5. **Most OpenAI/Cohere/sentence-transformers models**: cosine similarity recommended
6. **Many LLMs**: trained with dot product loss

---

## 13. Performance at Scale

### Performance by Vector Count

| Scale | Best Options | Notes |
|---|---|---|
| 1K-100K | ChromaDB, pgvector, FAISS, LanceDB | Any solution works. Optimize for developer experience |
| 100K-1M | pgvector, Qdrant, LanceDB, Redis | Single-node sufficient. ChromaDB still viable |
| 1M-10M | Qdrant, Milvus, Pinecone, Weaviate, pgvectorscale | Need production-grade indexing and filtering |
| 10M-100M | Qdrant, Milvus, Pinecone, Elasticsearch | Need horizontal scaling or managed service |
| 100M-1B | Milvus, Pinecone (DRN), Qdrant (distributed) | Distributed architecture essential |
| 1B+ | Milvus/Zilliz, Pinecone (DRN) | Purpose-built for billion-scale |

### Memory Requirements (Approximate, 768d, float32)

| Vector Count | Raw Vector Size | HNSW Index (M=16) | With Metadata |
|---|---|---|---|
| 100K | ~0.3 GB | ~0.5 GB | ~0.7 GB |
| 1M | ~3 GB | ~5 GB | ~7 GB |
| 10M | ~30 GB | ~50 GB | ~70 GB |
| 100M | ~300 GB | ~500 GB | ~700 GB |
| 1B | ~3 TB | ~5 TB | ~7 TB |

Quantization (int8, binary, PQ) can reduce these by 4-64x depending on technique.

### Latency Characteristics

| Operation | In-Memory (Redis) | HNSW (Qdrant/Milvus) | Disk-Based (LanceDB/pgvector) | Managed (Pinecone) |
|---|---|---|---|---|
| Simple query | <1ms | 1-5ms | 5-25ms | 20-50ms |
| Filtered query | 1-5ms | 5-65ms | 25-100ms | 30-80ms |
| Batch query (100) | 5-20ms | 20-100ms | 100-500ms | 50-200ms |
| Index build (1M) | Seconds | Minutes | Minutes | Managed |

---

## 14. Cost Comparison

### Monthly Cost Estimates by Scale (768d vectors)

| Scale | ChromaDB (self-hosted) | Pinecone (serverless) | Qdrant Cloud | Weaviate Cloud | Zilliz Cloud | pgvector (managed) |
|---|---|---|---|---|---|---|
| 100K vectors | ~$16 | ~$0 (free) | $0 (free) | $0 (trial) | $0 (free) | ~$15 |
| 1M vectors | ~$61 | ~$5-15 | ~$25 | ~$45 | ~$7-65 | ~$30 |
| 10M vectors | ~$243 | ~$50-150 | ~$100+ | ~$100+ | ~$70-650 | ~$100 |
| 100M vectors | N/A (scale limit) | ~$500-1,500 | ~$500+ | ~$400+ | ~$200-6,500 | ~$500+ |
| 1B vectors | N/A | Custom/DRN | Custom | Custom | ~$2,000-65,000 | N/A |

**Notes:**
- Costs vary dramatically based on query volume, not just storage
- Hidden costs: embeddings generation, reindexing, backups, data transfer
- Query cost scales with index size on Pinecone (1 RU per 1 GB namespace)
- Self-hosting at high, predictable query volume can cut costs 50-75%

### Cost Optimization Strategies

1. **Quantization**: Reduce memory 4-64x with scalar/binary/product quantization
2. **Tiered storage**: Hot/cold data separation (Zilliz, LanceDB)
3. **Dimension reduction**: Lower dimensions = proportionally lower costs
4. **Self-hosting**: 50-75% cheaper at high predictable volumes
5. **Namespace/partition pruning**: Query only relevant subsets
6. **Batch operations**: Reduce per-query overhead

---

## 15. Selection Decision Tree

### Quick Decision Framework

```
START
|
|-- Do you already use PostgreSQL?
|   |-- YES: Start with pgvector (+ pgvectorscale if >1M vectors)
|   |-- NO: Continue below
|
|-- Scale requirement?
|   |-- < 1M vectors (prototyping): ChromaDB or LanceDB
|   |-- 1M - 100M vectors: Continue below
|   |-- 100M+ vectors: Milvus/Zilliz or Pinecone DRN
|
|-- Do you need hybrid search (BM25 + vector)?
|   |-- YES: Weaviate, Elasticsearch, or Redis
|   |-- NO: Continue below
|
|-- Managed vs. self-hosted?
|   |-- Managed only: Pinecone
|   |-- Self-hosted preferred: Qdrant or Milvus
|   |-- Either: Any of the above
|
|-- Primary optimization target?
|   |-- Lowest latency: Redis (in-memory) or Qdrant
|   |-- Highest throughput: Qdrant or Milvus
|   |-- Lowest cost: LanceDB (embedded) or pgvector
|   |-- Multimodal data: LanceDB
|   |-- Operational simplicity: Pinecone or ChromaDB
|
|-- Multi-tenant SaaS?
|   |-- YES: Weaviate (native multi-tenancy) or Qdrant
|   |-- NO: Any
```

### Recommendation Matrix by Use Case

| Use Case | Primary Recommendation | Alternative |
|---|---|---|
| **RAG Prototype** | ChromaDB | LanceDB |
| **RAG Production (small)** | pgvector | Qdrant |
| **RAG Production (medium)** | Qdrant | Pinecone |
| **RAG Production (large)** | Milvus/Zilliz | Pinecone DRN |
| **Hybrid Search (text+vector)** | Weaviate | Elasticsearch |
| **Real-time Recommendations** | Redis | Qdrant |
| **Multimodal Search** | LanceDB | Weaviate |
| **Existing PostgreSQL Stack** | pgvector + pgvectorscale | Qdrant |
| **Existing Elasticsearch Stack** | Elasticsearch | Weaviate |
| **Multi-tenant SaaS** | Weaviate | Qdrant |
| **Edge/Embedded AI** | LanceDB | FAISS |
| **Maximum Performance** | Qdrant (CPU) / Milvus (GPU) | Redis (latency) |
| **Minimum Cost** | LanceDB or pgvector | ChromaDB |
| **Zero Ops** | Pinecone | Zilliz Cloud |
| **Custom Index Research** | FAISS | - |

### 2026 Industry Trend

There is a significant shift toward **extended relational databases** for vector search. PostgreSQL with pgvector/pgvectorscale is increasingly competitive, and many teams are consolidating their vector workloads into existing PostgreSQL infrastructure rather than adding a separate vector database. For RAG applications under 10M vectors with existing PostgreSQL infrastructure, pgvector is the pragmatic starting point.

---

## 16. Framework Integrations

### LangChain Integration Support

| Database | LangChain Integration | Status |
|---|---|---|
| ChromaDB | `langchain_chroma.Chroma` | First-class, most popular for prototyping |
| Pinecone | `langchain_pinecone.PineconeVectorStore` | First-class |
| Qdrant | `langchain_qdrant.QdrantVectorStore` | First-class |
| Weaviate | `langchain_weaviate.WeaviateVectorStore` | First-class |
| Milvus | `langchain_milvus.Milvus` | First-class |
| pgvector | `langchain_postgres.PGVector` | First-class |
| FAISS | `langchain_community.vectorstores.FAISS` | Built-in community |
| LanceDB | `langchain_community.vectorstores.LanceDB` | Community |
| Elasticsearch | `langchain_elasticsearch.ElasticsearchStore` | First-class |
| Redis | `langchain_redis.RedisVectorStore` | First-class |

### LlamaIndex Integration Support

| Database | LlamaIndex Integration | Status |
|---|---|---|
| ChromaDB | `llama_index.vector_stores.chroma` | First-class |
| Pinecone | `llama_index.vector_stores.pinecone` | First-class |
| Qdrant | `llama_index.vector_stores.qdrant` | First-class |
| Weaviate | `llama_index.vector_stores.weaviate` | First-class |
| Milvus | `llama_index.vector_stores.milvus` | First-class |
| pgvector | `llama_index.vector_stores.postgres` | First-class |
| FAISS | `llama_index.vector_stores.faiss` | First-class |
| LanceDB | `llama_index.vector_stores.lancedb` | First-class |
| Elasticsearch | `llama_index.vector_stores.elasticsearch` | First-class |
| Redis | `llama_index.vector_stores.redis` | First-class |

All 10 databases have first-class or community integrations with both LangChain and LlamaIndex. The integration quality is generally excellent across the board, with LangChain providing `VectorStore` abstractions and LlamaIndex providing `VectorStoreIndex` wrappers.

### Other Framework Support

| Database | Haystack | Semantic Kernel | Spring AI | DSPy |
|---|---|---|---|---|
| ChromaDB | Yes | Yes | Yes | Yes |
| Pinecone | Yes | Yes | Yes | Yes |
| Qdrant | Yes | Yes | Yes | Yes |
| Weaviate | Yes | Yes | Yes | Yes |
| Milvus | Yes | Yes | Partial | Yes |
| pgvector | Yes | Yes | Yes | Partial |
| FAISS | Partial | No | No | Yes |
| LanceDB | Partial | No | No | Partial |
| Elasticsearch | Yes | Yes | Yes | Partial |
| Redis | Yes | Yes | Yes | Partial |

---

## Summary: Key Takeaways

1. **No single best vector database** -- the right choice depends on scale, existing infrastructure, operational capacity, and budget.

2. **For RAG prototyping**: ChromaDB or LanceDB provide the fastest path to a working system.

3. **For production RAG**: Qdrant (self-hosted) or Pinecone (managed) are the most common choices in 2025-2026.

4. **For billion-scale**: Milvus/Zilliz is the most proven distributed solution.

5. **The pgvector trend**: Many teams are consolidating into PostgreSQL, especially under 10M vectors.

6. **Hybrid search is table stakes**: Weaviate, Elasticsearch, and Redis offer the most mature hybrid search; Pinecone, Qdrant, and Milvus have added it.

7. **Cost scales with queries, not just storage**: Understand your query patterns before committing to a pricing model.

8. **Quantization is critical**: 4-64x memory reduction with minimal recall loss. Every production deployment should use quantization.

9. **Run your own benchmarks**: Published benchmarks vary wildly based on configuration, hardware, and dataset. Always test with your actual data and query patterns.

10. **2026 convergence**: Traditional databases (PostgreSQL, Elasticsearch) are rapidly closing the gap with purpose-built vector databases, while purpose-built solutions are adding more traditional database features.
