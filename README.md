# ragfactory

Generate production-ready RAG pipelines from a single config.

```bash
pip install ragfactory

# Create a pipeline
ragfactory init --name my-rag --vector-db qdrant --llm anthropic --output ./my-rag

# Generate from an existing config
ragfactory generate --config pipeline.yaml --output ./my-rag

# Validate a config
ragfactory validate --config pipeline.yaml

# List available components
ragfactory options
ragfactory options --component embedding
```

## Quick-start

```yaml
# pipeline.yaml
name: my-pipeline
framework: langchain
indexing:
  chunking:
    type: recursive
  embedding:
    type: openai
  vector_db:
    type: qdrant
    url: http://localhost:6333
    collection_name: my-collection
retrieval:
  type: hybrid_rrf
generation:
  llm:
    type: anthropic
```

```bash
ragfactory generate --config pipeline.yaml --output ./my-pipeline
cd my-pipeline && pip install -r pyproject.toml
```

## Components

| Category   | Options |
|------------|---------|
| Chunking   | fixed, recursive, semantic, contextual, late, page_level, sentence_window |
| Embedding  | openai, cohere, voyage, gemini, bge_m3, nomic, jina |
| Vector DB  | chromadb, qdrant, pinecone, weaviate, milvus, pgvector |
| Retrieval  | dense, hybrid_rrf, hybrid_weighted, small_to_big, sentence_window |
| Reranker   | cohere, cross_encoder, colbert, flashrank |
| LLM        | openai, anthropic, cohere, ollama |

## Frameworks

Both **LangChain** and **LlamaIndex** are supported. Set `framework: langchain` or `framework: llamaindex` in your config.

## License

Apache-2.0
