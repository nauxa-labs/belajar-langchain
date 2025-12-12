---
sidebar_position: 4
title: Embeddings
description: Mengubah teks menjadi vector representations
---

# Embeddings

Embeddings mengubah teks menjadi **vector numerik** yang merepresentasikan makna semantik. Ini adalah fondasi dari semantic search dalam RAG.

## Bagaimana Embeddings Bekerja?

```
"Kucing adalah hewan peliharaan"
            │
            ▼
    [Embedding Model]
            │
            ▼
[0.023, -0.041, 0.089, -0.012, ...]  # 1536 dimensions untuk ada-002
```

**Properti penting:**
- Teks dengan **makna mirip** → vectors yang **dekat**
- Teks dengan **makna berbeda** → vectors yang **jauh**

```python
# "kucing" dan "cat" → vectors dekat
# "kucing" dan "mobil" → vectors jauh
```

## Similarity Measurement

### Cosine Similarity

Paling umum digunakan - mengukur sudut antar vectors.

```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# 1.0 = identik
# 0.0 = tidak terkait
# -1.0 = berlawanan (jarang)
```

### Euclidean Distance

Mengukur jarak langsung antar points.

```python
from numpy.linalg import norm

def euclidean_distance(a, b):
    return norm(a - b)

# Semakin kecil = semakin mirip
```

## OpenAI Embeddings

Paling populer dan high-quality.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # atau text-embedding-ada-002
)

# Embed single text
vector = embeddings.embed_query("Apa itu machine learning?")
print(f"Dimensions: {len(vector)}")  # 1536 atau 3072

# Embed multiple texts (batch)
vectors = embeddings.embed_documents([
    "Machine learning adalah cabang AI",
    "Python adalah bahasa pemrograman",
    "Kucing suka tidur"
])
print(f"Embedded {len(vectors)} documents")
```

### Models Available

| Model | Dimensions | Cost | Quality |
|-------|------------|------|---------|
| text-embedding-3-small | 1536 | $0.00002/1K tokens | Good |
| text-embedding-3-large | 3072 | $0.00013/1K tokens | Better |
| text-embedding-ada-002 | 1536 | $0.00010/1K tokens | Good (legacy) |

## Hugging Face Embeddings

Free dan bisa di-run locally.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("Hello world")
print(f"Dimensions: {len(vector)}")  # 384
```

```bash
pip install langchain-huggingface sentence-transformers
```

### Popular Models

| Model | Dimensions | Size | Language |
|-------|------------|------|----------|
| all-MiniLM-L6-v2 | 384 | 80MB | English |
| all-mpnet-base-v2 | 768 | 420MB | English |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 470MB | Multilingual |
| BAAI/bge-small-en-v1.5 | 384 | 130MB | English |
| BAAI/bge-m3 | 1024 | 2.2GB | Multilingual |

## Google Embeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vector = embeddings.embed_query("Test query")
```

## Ollama Embeddings (Local)

Run embeddings locally dengan Ollama.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"  # atau mxbai-embed-large
)

vector = embeddings.embed_query("Local embedding test")
```

```bash
# Install model dulu
ollama pull nomic-embed-text
```

## Cohere Embeddings

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-english-v3.0"  # atau embed-multilingual-v3.0
)

# Cohere has input types
vector = embeddings.embed_query("search query")  # Untuk queries
vectors = embeddings.embed_documents(["doc1", "doc2"])  # Untuk documents
```

## Semantic Search Demo

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings()

# Documents
documents = [
    "Python adalah bahasa pemrograman yang mudah dipelajari",
    "JavaScript digunakan untuk web development",
    "Machine learning membutuhkan banyak data",
    "Kucing adalah hewan peliharaan yang populer",
    "Neural networks terinspirasi dari otak manusia"
]

# Embed documents
doc_vectors = embeddings.embed_documents(documents)

# Query
query = "Bagaimana cara belajar coding?"
query_vector = embeddings.embed_query(query)

# Calculate similarities
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [
    (doc, cosine_similarity(query_vector, vec))
    for doc, vec in zip(documents, doc_vectors)
]

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

print(f"Query: {query}\n")
for doc, score in similarities:
    print(f"{score:.4f}: {doc}")
```

Output:
```
Query: Bagaimana cara belajar coding?

0.8234: Python adalah bahasa pemrograman yang mudah dipelajari
0.7891: JavaScript digunakan untuk web development
0.6543: Machine learning membutuhkan banyak data
0.6234: Neural networks terinspirasi dari otak manusia
0.4123: Kucing adalah hewan peliharaan yang populer
```

## Caching Embeddings

Untuk hemat API calls dan biaya.

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace=OpenAIEmbeddings().model
)

# First call - hits API
vector1 = cached_embeddings.embed_query("Test")

# Second call - from cache (instant!)
vector2 = cached_embeddings.embed_query("Test")
```

## Dimensionality Reduction

Kadang perlu reduce dimensi untuk performance.

```python
from langchain_openai import OpenAIEmbeddings

# text-embedding-3-small supports dimension reduction
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=256  # Reduce from 1536 to 256
)
```

## Choosing the Right Model

### Factors to Consider

1. **Quality vs Cost**
   - OpenAI: Best quality, pay per use
   - HuggingFace: Free, good quality
   - Ollama: Free, local, privacy

2. **Language Support**
   - English only: all-MiniLM, BGE-en
   - Multilingual: BGE-m3, multilingual-MiniLM

3. **Dimensions**
   - Higher = more accurate, more storage
   - Lower = faster, less storage

4. **Latency**
   - API: Network latency
   - Local: No latency but slower inference

### Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Production English | OpenAI text-embedding-3-small |
| Production Multilingual | Cohere embed-multilingual |
| Development/Testing | HuggingFace all-MiniLM-L6-v2 |
| Privacy-sensitive | Ollama nomic-embed-text |
| Cost-sensitive | HuggingFace BGE models |

## Best Practices

### 1. Batch Embedding

```python
# ❌ Slow - one at a time
for doc in documents:
    vector = embeddings.embed_query(doc)

# ✅ Fast - batch
vectors = embeddings.embed_documents(documents)
```

### 2. Use Same Model for Query & Docs

```python
# ❌ Wrong - different models
doc_embeddings = OpenAIEmbeddings()
query_embeddings = HuggingFaceEmbeddings()

# ✅ Correct - same model
embeddings = OpenAIEmbeddings()
doc_vectors = embeddings.embed_documents(docs)
query_vector = embeddings.embed_query(query)
```

### 3. Preprocess Text

```python
def preprocess(text):
    # Remove extra whitespace
    text = " ".join(text.split())
    # Lowercase (optional, depends on model)
    # text = text.lower()
    return text

clean_docs = [preprocess(doc) for doc in documents]
vectors = embeddings.embed_documents(clean_docs)
```

### 4. Monitor Token Usage

```python
import tiktoken

def count_tokens(text, model="cl100k_base"):
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

total_tokens = sum(count_tokens(doc) for doc in documents)
estimated_cost = total_tokens * 0.00002 / 1000  # text-embedding-3-small
print(f"Estimated cost: ${estimated_cost:.4f}")
```

## Ringkasan

1. **Embeddings** = text → vector representation
2. **Cosine similarity** untuk mengukur kesamaan
3. **OpenAI** untuk production, **HuggingFace/Ollama** untuk development
4. **Batch embedding** untuk efficiency
5. **Cache** untuk hemat biaya
6. **Same model** untuk query dan documents

---

**Selanjutnya:** [Vector Stores](/docs/rag/vector-stores) - Menyimpan dan mencari embeddings.
