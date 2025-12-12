---
sidebar_position: 1
title: Konsep RAG
description: Memahami Retrieval Augmented Generation dan mengapa ini penting
---

# Konsep RAG

RAG (Retrieval Augmented Generation) adalah teknik yang menggabungkan **retrieval** (pencarian informasi) dengan **generation** (pembuatan teks oleh LLM). Ini adalah salah satu pattern paling penting dalam aplikasi LLM.

## Masalah yang Diselesaikan RAG

### 1. Knowledge Cutoff

LLM memiliki "knowledge cutoff" - tidak tahu informasi setelah tanggal training.

```
User: "Siapa presiden Indonesia saat ini?"
LLM (trained 2023): "Joko Widodo"  # Mungkin sudah outdated!
```

### 2. Hallucination

LLM bisa menghasilkan informasi yang terdengar benar tapi salah.

```
User: "Apa kebijakan cuti di perusahaan kami?"
LLM: "Karyawan mendapat 14 hari cuti per tahun..."  # MADE UP!
```

### 3. Tidak Punya Data Private

LLM tidak tahu tentang:
- Dokumen internal perusahaan
- Database produk Anda
- Knowledge base khusus

## Solusi: RAG

RAG mengatasi masalah ini dengan **memberikan context relevan ke LLM**.

```
┌─────────────────────────────────────────────────────────┐
│                         RAG Flow                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Query ──▶ [Retriever] ──▶ Relevant Documents      │
│                      │                   │               │
│                      ▼                   ▼               │
│              Knowledge Base        [LLM + Context]       │
│              (Vector Store)              │               │
│                                          ▼               │
│                                    Final Answer          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Contoh dengan RAG

```python
# Tanpa RAG
prompt = "Apa kebijakan cuti di perusahaan?"
# LLM hanya bisa guess/hallucinate

# Dengan RAG
context = retrieve_from_knowledge_base("kebijakan cuti")
# context = "Menurut Handbook v3.2: Karyawan tetap mendapat 21 hari cuti..."

prompt = f"""
Berdasarkan dokumen berikut:
{context}

Pertanyaan: Apa kebijakan cuti di perusahaan?
"""
# LLM menjawab berdasarkan dokumen ASLI
```

## Komponen RAG

### 1. Document Loaders

Membaca dokumen dari berbagai sumber.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader
)

# Load PDF
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()

# Load from web
loader = WebBaseLoader("https://example.com/docs")
docs = loader.load()
```

### 2. Text Splitters

Memecah dokumen menjadi chunks yang lebih kecil.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

**Mengapa perlu di-split?**
- LLM memiliki context window terbatas
- Chunks kecil = retrieval lebih precise
- Overlap memastikan konteks tidak terpotong

### 3. Embeddings

Mengubah teks menjadi vector numerik.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Text → Vector
vector = embeddings.embed_query("Apa kebijakan cuti?")
# [0.023, -0.041, 0.089, ...]  # 1536 dimensions
```

**Bagaimana embeddings bekerja?**
- Teks dengan makna mirip → vectors yang dekat
- "King - Man + Woman ≈ Queen" (klasik word2vec)
- Memungkinkan semantic search

### 4. Vector Store

Menyimpan dan mencari vectors.

```python
from langchain_chroma import Chroma

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Search
results = vectorstore.similarity_search("kebijakan cuti", k=3)
```

**Vector stores populer:**
- **Chroma** - Simple, local, great for dev
- **FAISS** - Facebook's library, very fast
- **Pinecone** - Fully managed, production-ready
- **Weaviate** - Open source, feature-rich
- **Qdrant** - High performance, Rust-based

### 5. Retriever

Interface untuk mengambil dokumen relevan.

```python
# Convert vector store to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Retrieve
relevant_docs = retriever.invoke("kebijakan cuti")
```

### 6. RAG Chain

Menggabungkan semua komponen.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer based on the following context:

{context}

Question: {input}
""")

# Create chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Use
result = rag_chain.invoke({"input": "Apa kebijakan cuti?"})
print(result["answer"])
```

## RAG vs Fine-tuning

| Aspek | RAG | Fine-tuning |
|-------|-----|-------------|
| **Knowledge update** | Mudah (update docs) | Sulit (retrain) |
| **Cost** | Rendah | Tinggi |
| **Accuracy** | Factual, verifiable | Bisa hallucinate |
| **Customization** | Content-based | Behavior-based |
| **Best for** | Knowledge retrieval | Style/format changes |

## Kapan Pakai RAG?

✅ **Gunakan RAG untuk:**
- Q&A dari dokumen
- Customer support dengan knowledge base
- Chatbot dengan data terkini
- Search & summarization

❌ **Tidak cocok untuk:**
- Task yang tidak butuh external knowledge
- Mengubah "personality" LLM
- Real-time data (perlu integrasi API)

## RAG Architecture Patterns

### 1. Basic RAG

```
Query → Retrieve → Generate
```

### 2. RAG with Reranking

```
Query → Retrieve (many) → Rerank → Generate (top-k)
```

### 3. Multi-query RAG

```
Query → Generate variations → Retrieve each → Merge → Generate
```

### 4. Agentic RAG

```
Query → Agent decides → Retrieve/Tool → Maybe iterate → Generate
```

## Ringkasan

1. **RAG** = Retrieval + Generation
2. **Solves** knowledge cutoff, hallucination, private data
3. **Components**: Loaders → Splitters → Embeddings → Vector Store → Retriever
4. **Better than fine-tuning** untuk knowledge-based tasks
5. Berbagai **patterns** untuk use cases berbeda

---

**Selanjutnya:** [Document Loaders](/docs/rag/document-loaders) - Membaca dokumen dari berbagai sumber.
