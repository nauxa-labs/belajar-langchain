---
sidebar_position: 6
title: Retrievers
description: Interface untuk retrieval dan advanced patterns
---

# Retrievers

Retriever adalah **interface abstrak** untuk mengambil dokumen relevan. Ini memisahkan logika retrieval dari vector store sehingga bisa diganti dan dikombinasikan.

## Basic Retriever

### From Vector Store

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# Convert to retriever
retriever = vectorstore.as_retriever()

# Use
relevant_docs = retriever.invoke("What is machine learning?")
```

### Retriever Interface

Semua retrievers memiliki interface yang sama:

```python
# Sync
docs = retriever.invoke(query)

# Async
docs = await retriever.ainvoke(query)

# Batch
docs_list = retriever.batch([query1, query2, query3])
```

## Configuring Retrievers

### Search Type

```python
# Similarity search (default)
retriever = vectorstore.as_retriever(
    search_type="similarity"
)

# MMR - Maximum Marginal Relevance (diverse results)
retriever = vectorstore.as_retriever(
    search_type="mmr"
)

# Similarity with threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8}
)
```

### Search Parameters

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,                    # Number of results
        "filter": {"source": "handbook.pdf"},  # Metadata filter
        "fetch_k": 20,             # For MMR: fetch before selecting
        "lambda_mult": 0.5         # For MMR: diversity factor
    }
)
```

## Multi-Query Retriever

Generate multiple query variations untuk better recall.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Single query → multiple variations → retrieve from each → deduplicate
docs = multi_retriever.invoke("What are the effects of climate change?")
```

### How It Works

```
Original Query: "What are the effects of climate change?"
        │
        ▼ (LLM generates variations)
        │
    ┌───┴───────────────────────────────────────┐
    │                                           │
"How does climate      "What environmental    "Climate change
 change impact          problems result        consequences"
 the environment?"      from global warming?"
    │                           │                    │
    ▼                           ▼                    ▼
 [Retrieve]                 [Retrieve]           [Retrieve]
    │                           │                    │
    └───────────────┬───────────┴────────────────────┘
                    │
                    ▼
              [Deduplicate]
                    │
                    ▼
               Final Docs
```

## Contextual Compression

Compress retrieved documents to only include relevant parts.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create compressor
compressor = LLMChainExtractor.from_llm(llm)

# Wrap retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# Returns compressed, relevant portions
docs = compression_retriever.invoke("What is the refund policy?")
```

### Other Compressors

```python
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline
)
from langchain_text_splitters import CharacterTextSplitter

# Filter by embedding similarity
embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76
)

# Pipeline: split → filter
splitter = CharacterTextSplitter(chunk_size=300)
pipeline = DocumentCompressorPipeline(
    transformers=[splitter, embeddings_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=vectorstore.as_retriever()
)
```

## Ensemble Retriever

Combine multiple retrievers dengan weighted scoring.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Semantic retriever
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Keyword retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 60% semantic, 40% keyword
)

docs = ensemble_retriever.invoke("machine learning algorithms")
```

### Why Ensemble?

- **Semantic search** bagus untuk meaning/concept
- **Keyword search** bagus untuk exact terms, names, codes
- **Combine** untuk best of both worlds

## Self-Query Retriever

Automatically parse query into semantic query + metadata filter.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

# Define metadata attributes
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="The year the document was published",
        type="integer"
    ),
    AttributeInfo(
        name="department",
        description="The department that created the document",
        type="string"
    ),
]

# Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectorstore,
    document_contents="Company policy documents",
    metadata_field_info=metadata_field_info
)

# Natural language query → structured query
docs = self_query_retriever.invoke("Show me HR policies from 2024")
# Automatically applies filter: {department: "HR", year: 2024}
```

## Parent Document Retriever

Retrieve small chunks, return parent documents.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Parent splitter (large chunks)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Child splitter (small chunks for retrieval)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Storage for parent docs
store = InMemoryStore()

# Create retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Add documents
retriever.add_documents(docs)

# Search returns PARENT documents (more context)
results = retriever.invoke("specific query")
```

### How It Works

```
Original Document
        │
        ▼
┌──────────────────────────────────┐
│        Parent Chunk (2000)        │
│  ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Child1│ │Child2│ │Child3│      │
│  │(400) │ │(400) │ │(400) │      │
│  └──────┘ └──────┘ └──────┘     │
└──────────────────────────────────┘

Query matches Child2 → Return Parent Chunk (more context!)
```

## Time-Weighted Retriever

Prioritize recent documents.

```python
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from datetime import datetime

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,  # How fast old docs lose relevance
    k=5
)

# Add with timestamp
retriever.add_documents([
    Document(
        page_content="New policy update",
        metadata={"last_accessed_at": datetime.now()}
    )
])
```

## Custom Retriever

Buat retriever sendiri untuk logic khusus.

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class CustomRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 5
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Custom logic
        docs = self.vectorstore.similarity_search(query, k=self.k * 2)
        
        # Filter/rerank
        filtered = [d for d in docs if len(d.page_content) > 100]
        
        return filtered[:self.k]

retriever = CustomRetriever(vectorstore=vectorstore, k=3)
docs = retriever.invoke("my query")
```

## Retriever in LCEL Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
Answer based on context:

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is the vacation policy?")
```

## Comparison

| Retriever | Use Case | Overhead |
|-----------|----------|----------|
| Basic | Simple Q&A | Low |
| Multi-Query | Complex queries | Medium (LLM calls) |
| Ensemble | Hybrid search | Low |
| Self-Query | Natural language filters | Medium (LLM call) |
| Parent Document | Need more context | Low |
| Compression | Long docs | High (LLM per doc) |

## Ringkasan

1. **as_retriever()** - convert vector store to retriever
2. **Multi-Query** - generate query variations for recall
3. **Ensemble** - combine semantic + keyword search  
4. **Self-Query** - parse filters from natural language
5. **Parent Document** - retrieve context, not just chunks
6. **Compression** - extract only relevant portions
7. Semua retrievers dapat di-chain dengan **LCEL**

---

**Selanjutnya:** [Basic RAG Chain](/docs/rag/basic-rag-chain) - Membangun pipeline RAG lengkap.
