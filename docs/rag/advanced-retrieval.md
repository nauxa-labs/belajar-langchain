---
sidebar_position: 8
title: Advanced Retrieval
description: Teknik retrieval lanjutan - hybrid search, HyDE, reranking
---

# Advanced Retrieval

Teknik retrieval lanjutan untuk meningkatkan akurasi dan relevansi dokumen yang diambil.

## Hybrid Search

Menggabungkan **semantic search** (embeddings) dengan **keyword search** (BM25).

### Kenapa Hybrid?

| Search Type | Good For | Bad For |
|-------------|----------|---------|
| Semantic | Concepts, paraphrases | Exact terms, codes |
| Keyword | IDs, names, acronyms | Variations, synonyms |
| **Hybrid** | Both! | More complex |

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Documents
docs = [...]  # Your documents

# Semantic retriever
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Keyword retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# Combine
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # Tune based on use case
)

# Use
results = hybrid_retriever.invoke("policy ID-2024-001 about remote work")
```

### Weight Tuning Guidelines

| Use Case | Semantic Weight | Keyword Weight |
|----------|-----------------|----------------|
| General Q&A | 0.7 | 0.3 |
| Technical docs | 0.5 | 0.5 |
| Legal/exact terms | 0.3 | 0.7 |
| Code search | 0.4 | 0.6 |

## HyDE (Hypothetical Document Embeddings)

Generate hypothetical answer first, then search for similar documents.

### Concept

```
Query: "What is the refund policy?"
        │
        ▼ (LLM generates hypothetical answer)
        │
"Our refund policy allows customers to return 
 products within 30 days for a full refund..."
        │
        ▼ (Embed this hypothetical document)
        │
[0.023, -0.041, ...]
        │
        ▼ (Search vector store)
        │
Actual documents about refund policy
```

### Implementation

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Create HyDE embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
base_embeddings = OpenAIEmbeddings()

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"  # or custom prompt
)

# Use for indexing or querying
query_vector = hyde_embeddings.embed_query("What is the vacation policy?")
```

### Custom HyDE Prompt

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

hyde_prompt = ChatPromptTemplate.from_template("""
Kamu adalah expert HR yang menulis kebijakan perusahaan.
Tulis paragraf singkat yang menjawab pertanyaan ini seolah-olah kamu sedang menulis dokumen kebijakan resmi.

Pertanyaan: {question}

Dokumen kebijakan:
""")

# Manual HyDE
def hyde_search(question: str, vectorstore, llm):
    # Generate hypothetical document
    hypothetical = (hyde_prompt | llm | StrOutputParser()).invoke({"question": question})
    
    # Search using hypothetical as query
    results = vectorstore.similarity_search(hypothetical, k=5)
    
    return results
```

## Reranking

Retrieve many, then rerank to get most relevant.

### Cross-Encoder Reranking

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

# Cross-encoder model
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=model, top_n=3)

# Wrap base retriever
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})  # Fetch more
)

# Results are reranked
docs = reranking_retriever.invoke("What is machine learning?")
```

### Cohere Reranking

```python
from langchain_cohere import CohereRerank

reranker = CohereRerank(
    model="rerank-english-v3.0",  # or rerank-multilingual
    top_n=5
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)
```

### LLM-based Reranking

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List

class RankedDocs(BaseModel):
    rankings: List[int]  # Indices of docs in order of relevance

rerank_prompt = ChatPromptTemplate.from_template("""
Given this query and documents, rank the documents by relevance.
Return the indices (0-based) in order of most to least relevant.

Query: {query}

Documents:
{documents}

Return only the indices as a list, e.g., [2, 0, 3, 1]
""")

def llm_rerank(query: str, docs: list, llm, top_k: int = 3):
    docs_text = "\n\n".join([
        f"[{i}] {doc.page_content[:500]}" 
        for i, doc in enumerate(docs)
    ])
    
    chain = rerank_prompt | llm.with_structured_output(RankedDocs)
    result = chain.invoke({"query": query, "documents": docs_text})
    
    return [docs[i] for i in result.rankings[:top_k]]
```

## Multi-Vector Retriever

Store multiple vectors per document for better retrieval.

### Summary + Full Document

```python
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

# Stores
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()

# Create retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id"
)

# For each document, create summary
llm = ChatOpenAI(model="gpt-4o-mini")

for doc in documents:
    doc_id = str(uuid.uuid4())
    
    # Create summary
    summary = llm.invoke(f"Summarize this:\n{doc.page_content}").content
    
    # Add summary to vectorstore (for retrieval)
    summary_doc = Document(
        page_content=summary,
        metadata={"doc_id": doc_id}
    )
    vectorstore.add_documents([summary_doc])
    
    # Add full doc to docstore (to return)
    docstore.mset([(doc_id, doc)])

# Search finds summaries, returns full docs
results = retriever.invoke("query")
```

### Questions + Document

```python
# Generate questions per document
for doc in documents:
    doc_id = str(uuid.uuid4())
    
    # Generate questions
    questions = llm.invoke(
        f"Generate 3 questions this document answers:\n{doc.page_content}"
    ).content.split("\n")
    
    # Add questions to vectorstore
    for q in questions:
        q_doc = Document(page_content=q, metadata={"doc_id": doc_id})
        vectorstore.add_documents([q_doc])
    
    # Add full doc to docstore
    docstore.mset([(doc_id, doc)])
```

## Recursive Retrieval

Retrieve, then retrieve again based on results.

```python
from langchain_core.runnables import RunnableLambda

def recursive_retrieve(query: str, depth: int = 2):
    all_docs = []
    current_query = query
    
    for i in range(depth):
        docs = retriever.invoke(current_query)
        all_docs.extend(docs)
        
        # Extract key terms for next query
        if docs:
            key_content = docs[0].page_content[:200]
            current_query = f"{query} {key_content}"
    
    # Deduplicate by content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs
```

## Query Transformation

### Query Expansion

```python
from langchain_core.prompts import ChatPromptTemplate

expand_prompt = ChatPromptTemplate.from_template("""
Generate 3 alternative ways to ask this question:

Original: {question}

Alternatives (one per line):
""")

def expand_query(question: str, llm):
    result = (expand_prompt | llm | StrOutputParser()).invoke({"question": question})
    alternatives = result.strip().split("\n")
    return [question] + alternatives

# Search all variations, merge results
def expanded_search(question: str, retriever, llm):
    queries = expand_query(question, llm)
    
    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)
    
    # Deduplicate and return
    return list({doc.page_content: doc for doc in all_docs}.values())
```

### Step-back Prompting

Ask a more general question first.

```python
stepback_prompt = ChatPromptTemplate.from_template("""
Given this specific question, generate a more general/abstract question that would help answer it.

Specific: {question}

General question:
""")

def stepback_search(question: str, retriever, llm):
    # Get general question
    general = (stepback_prompt | llm | StrOutputParser()).invoke({"question": question})
    
    # Search both
    specific_docs = retriever.invoke(question)
    general_docs = retriever.invoke(general)
    
    # Combine and deduplicate
    all_docs = specific_docs + general_docs
    return list({doc.page_content: doc for doc in all_docs}.values())
```

## Comparison

| Technique | When to Use | Overhead |
|-----------|-------------|----------|
| Hybrid Search | Mixed query types | Low |
| HyDE | Query-document gap | Medium (1 LLM call) |
| Reranking | Need precision | Medium-High |
| Multi-Vector | Long documents | High (storage) |
| Query Expansion | Recall important | Medium (LLM calls) |

## Ringkasan

1. **Hybrid Search** - combine semantic + keyword
2. **HyDE** - generate hypothetical answer first  
3. **Reranking** - fetch many, score with cross-encoder
4. **Multi-Vector** - multiple representations per doc
5. **Query Transformation** - expand or abstract queries

---

**Selanjutnya:** [RAG Evaluation](/docs/rag/rag-evaluation) - Mengukur dan meningkatkan kualitas RAG.
