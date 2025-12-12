---
sidebar_position: 5
title: Vector Stores
description: Menyimpan dan mencari embeddings dengan vector databases
---

# Vector Stores

Vector Stores adalah database khusus untuk menyimpan dan mencari vectors (embeddings). Ini adalah **komponen kritis** dalam RAG untuk retrieval.

## Konsep Dasar

```
┌─────────────────────────────────────────────────────────┐
│                     Vector Store                         │
├─────────────────────────────────────────────────────────┤
│  ID   │     Vector                    │  Metadata       │
├───────┼───────────────────────────────┼─────────────────┤
│  doc1 │ [0.023, -0.041, 0.089, ...]  │ {source: "a.pdf"}│
│  doc2 │ [0.067, 0.012, -0.034, ...]  │ {source: "b.pdf"}│
│  doc3 │ [-0.045, 0.078, 0.023, ...]  │ {source: "c.pdf"}│
└───────┴───────────────────────────────┴─────────────────┘

Query: "Apa kebijakan cuti?"
        │
        ▼
[0.015, -0.038, 0.092, ...]  (query vector)
        │
        ▼  (similarity search)
        │
doc2 (most similar) ──▶ Return
```

## Chroma

Simple, in-memory atau persistent. **Great untuk development**.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection"
)

# Search
results = vectorstore.similarity_search("query", k=3)
```

### Persistent Storage

```python
# Save to disk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Load later
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### Installation

```bash
pip install langchain-chroma
```

## FAISS

Facebook AI Similarity Search. **Sangat cepat**, cocok untuk jutaan vectors.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Search
results = vectorstore.similarity_search("query", k=5)
```

### Save & Load

```python
# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

### Installation

```bash
pip install faiss-cpu  # atau faiss-gpu untuk GPU
```

## Pinecone

Fully managed, production-ready. **Best untuk production scale**.

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="your-api-key")

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# Or connect to existing
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="my-index",
    embedding=embeddings
)
```

### Installation

```bash
pip install langchain-pinecone pinecone-client
```

## Qdrant

High-performance, feature-rich. **Great balance of features**.

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Local
client = QdrantClient(path="./qdrant_db")

# Or Cloud
client = QdrantClient(
    url="https://xxx.qdrant.io",
    api_key="your-api-key"
)

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    collection_name="my_docs"
)
```

### Installation

```bash
pip install langchain-qdrant qdrant-client
```

## Weaviate

Open source dengan GraphQL interface.

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.connect_to_local()  # atau connect_to_wcs

vectorstore = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    index_name="MyDocs"
)
```

## PostgreSQL (pgvector)

Postgres dengan vector extension. **Jika sudah pakai Postgres**.

```python
from langchain_postgres.vectorstores import PGVector

connection_string = "postgresql://user:password@localhost:5432/vectordb"

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection=connection_string,
    collection_name="my_docs"
)
```

```bash
pip install langchain-postgres
```

## Search Methods

### Similarity Search

Default - cari yang paling mirip.

```python
results = vectorstore.similarity_search(
    query="machine learning",
    k=5  # Return top 5
)

for doc in results:
    print(doc.page_content[:100])
    print(doc.metadata)
```

### Similarity Search with Score

Get similarity scores.

```python
results = vectorstore.similarity_search_with_score(
    query="machine learning",
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(doc.page_content[:100])
```

### Maximum Marginal Relevance (MMR)

Diversify results - avoid returning similar documents.

```python
results = vectorstore.max_marginal_relevance_search(
    query="machine learning",
    k=5,
    fetch_k=20,  # Fetch 20, then select 5 diverse ones
    lambda_mult=0.5  # 0=max diversity, 1=max relevance
)
```

## Filtering

Filter berdasarkan metadata.

### Chroma Filtering

```python
results = vectorstore.similarity_search(
    query="kebijakan",
    k=5,
    filter={"source": "handbook.pdf"}
)

# Multiple conditions
results = vectorstore.similarity_search(
    query="kebijakan",
    k=5,
    filter={
        "$and": [
            {"source": "handbook.pdf"},
            {"department": "HR"}
        ]
    }
)
```

### FAISS Filtering

FAISS tidak support native filtering, filter setelah search:

```python
results = vectorstore.similarity_search("query", k=20)
filtered = [doc for doc in results if doc.metadata.get("year") == 2024][:5]
```

### Pinecone Filtering

```python
results = vectorstore.similarity_search(
    query="policy",
    k=5,
    filter={
        "department": {"$eq": "HR"},
        "year": {"$gte": 2023}
    }
)
```

## Adding Documents

### Add to Existing Store

```python
# Add new documents
new_docs = [Document(page_content="New content", metadata={"source": "new.txt"})]
vectorstore.add_documents(new_docs)

# Add with IDs
vectorstore.add_documents(new_docs, ids=["doc_001", "doc_002"])
```

### Add Texts Directly

```python
texts = ["Text 1", "Text 2", "Text 3"]
metadatas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]

vectorstore.add_texts(texts, metadatas=metadatas)
```

## Deleting Documents

```python
# Delete by IDs
vectorstore.delete(ids=["doc_001", "doc_002"])

# Delete by filter (if supported)
vectorstore.delete(filter={"source": "old.pdf"})
```

## As Retriever

Convert vector store ke retriever untuk use dalam chains.

```python
# Basic retriever
retriever = vectorstore.as_retriever()

# With configuration
retriever = vectorstore.as_retriever(
    search_type="similarity",  # atau "mmr"
    search_kwargs={
        "k": 5,
        "filter": {"department": "Engineering"}
    }
)

# Use in chain
relevant_docs = retriever.invoke("query")
```

## Comparison Table

| Vector Store | Persistence | Filtering | Managed | Best For |
|--------------|-------------|-----------|---------|----------|
| Chroma | Local file | ✅ | No | Development, small apps |
| FAISS | Local file | ❌ | No | Speed, large datasets |
| Pinecone | Cloud | ✅ | Yes | Production, scale |
| Qdrant | Both | ✅ | Both | Production, features |
| Weaviate | Both | ✅ | Both | GraphQL, modules |
| pgvector | Postgres | ✅ | No | Existing Postgres users |

## Complete Example

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. Embed & Store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./handbook_db"
)

# 4. Search
results = vectorstore.similarity_search_with_score(
    query="Berapa hari cuti per tahun?",
    k=3
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Source: {doc.metadata.get('source')}")
    print(doc.page_content[:200])
    print("---")
```

## Best Practices

### 1. Choose Based on Scale

```python
# Development: Chroma
# Small production: FAISS
# Large production: Pinecone/Qdrant
```

### 2. Add Meaningful Metadata

```python
doc = Document(
    page_content="Content...",
    metadata={
        "source": "handbook.pdf",
        "page": 5,
        "department": "HR",
        "last_updated": "2024-01-15",
        "doc_type": "policy"
    }
)
```

### 3. Index Management

```python
# Create separate indices for different use cases
hr_vectorstore = Chroma(collection_name="hr_docs", ...)
engineering_vectorstore = Chroma(collection_name="eng_docs", ...)
```

### 4. Monitor Performance

```python
import time

start = time.time()
results = vectorstore.similarity_search(query, k=5)
latency = time.time() - start

print(f"Search latency: {latency*1000:.0f}ms")
```

## Ringkasan

1. **Vector stores** menyimpan embeddings untuk similarity search
2. **Chroma** untuk development, **Pinecone/Qdrant** untuk production
3. **Similarity search** untuk basic, **MMR** untuk diversity
4. **Filtering** dengan metadata untuk precise retrieval
5. **as_retriever()** untuk integrasi dengan chains
6. Pilih berdasarkan **scale**, **features**, dan **infrastructure**

---

**Selanjutnya:** [Retrievers](/docs/rag/retrievers) - Interface untuk retrieval dan advanced patterns.
