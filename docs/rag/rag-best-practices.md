---
sidebar_position: 10
title: RAG Best Practices
description: Tips, patterns, dan use case untuk production RAG
---

# RAG Best Practices

Kumpulan best practices dan patterns untuk membangun RAG system yang robust di production.

## Indexing Best Practices

### 1. Optimal Chunk Size

```python
# Testing chunk sizes
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Test different sizes
chunk_sizes = [500, 1000, 1500, 2000]

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.2)  # 20% overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Size {size}: {len(chunks)} chunks, avg {sum(len(c.page_content) for c in chunks)/len(chunks):.0f} chars")
```

**Guidelines:**
| Content Type | Recommended Size | Overlap |
|--------------|------------------|---------|
| Q&A / FAQ | 500-800 | 100-150 |
| Technical docs | 1000-1500 | 200-300 |
| Legal documents | 800-1200 | 150-250 |
| Conversational | 300-600 | 50-100 |

### 2. Rich Metadata

```python
from datetime import datetime

def enrich_metadata(doc, source_info):
    """Add rich metadata for filtering and context."""
    doc.metadata.update({
        # Source info
        "source": source_info.get("filename"),
        "source_type": source_info.get("type", "document"),
        
        # Temporal
        "indexed_at": datetime.now().isoformat(),
        "last_modified": source_info.get("modified_date"),
        
        # Organizational
        "department": source_info.get("department"),
        "document_type": source_info.get("doc_type"),
        "access_level": source_info.get("access_level", "public"),
        
        # Content hints
        "language": detect_language(doc.page_content),
        "word_count": len(doc.page_content.split())
    })
    return doc
```

### 3. Preprocessing Pipeline

```python
import re

def preprocess_text(text: str) -> str:
    """Clean and normalize text before indexing."""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Strip leading/trailing
    text = text.strip()
    
    return text

# Apply during chunking
chunks = splitter.split_documents(docs)
for chunk in chunks:
    chunk.page_content = preprocess_text(chunk.page_content)
```

## Retrieval Best Practices

### 1. Hybrid Search by Default

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def create_hybrid_retriever(vectorstore, docs, semantic_weight=0.6):
    """Create production-ready hybrid retriever."""
    
    semantic = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    keyword = BM25Retriever.from_documents(docs)
    keyword.k = 10
    
    return EnsembleRetriever(
        retrievers=[semantic, keyword],
        weights=[semantic_weight, 1 - semantic_weight]
    )
```

### 2. Two-Stage Retrieval

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def create_two_stage_retriever(vectorstore):
    """Retrieve many, then rerank to get best."""
    
    # Stage 1: Broad retrieval
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}  # Get more
    )
    
    # Stage 2: Rerank
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
    
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
```

### 3. Metadata Filtering

```python
def filtered_retriever(vectorstore, user_context: dict):
    """Create retriever with user-specific filters."""
    
    filter_dict = {}
    
    # Filter by access level
    if user_context.get("access_level"):
        filter_dict["access_level"] = user_context["access_level"]
    
    # Filter by department
    if user_context.get("department"):
        filter_dict["department"] = user_context["department"]
    
    # Filter by recency
    if user_context.get("only_recent"):
        filter_dict["year"] = {"$gte": 2023}
    
    return vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": filter_dict}
    )
```

## Generation Best Practices

### 1. Structured Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions based on company documents.

RULES:
1. Answer ONLY based on the provided context
2. If information is not in context, say "I couldn't find this information"
3. Cite sources when possible using [Source: filename]
4. Be concise but complete
5. If asked about multiple topics, address each one"""),
    
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])
```

### 2. Source Attribution

```python
def format_docs_with_sources(docs):
    """Format documents with clear source attribution."""
    formatted = []
    
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        
        source_line = f"[Source {i+1}: {source}"
        if page:
            source_line += f", Page {page}"
        source_line += "]"
        
        formatted.append(f"{source_line}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)
```

### 3. Answer with Confidence

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence based on context quality"
    )
    sources: List[str] = Field(description="List of source documents used")
    caveats: str = Field(default="", description="Any limitations or caveats")

# Use structured output
rag_chain = prompt | llm.with_structured_output(RAGAnswer)
```

## Error Handling

### 1. Graceful Fallbacks

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import logging

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def safe_rag_chain(retriever, llm, prompt):
    """RAG chain with error handling."""
    
    def safe_retrieve(query):
        try:
            docs = retriever.invoke(query)
            if not docs:
                return [Document(page_content="No relevant documents found.")]
            return docs
        except Exception as e:
            logging.error(f"Retrieval error: {e}")
            return [Document(page_content="Error retrieving documents.")]
    
    def safe_generate(inputs):
        try:
            return (prompt | llm | StrOutputParser()).invoke(inputs)
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return "I apologize, but I'm unable to generate a response right now."
    
    return (
        {"context": RunnableLambda(safe_retrieve) | format_docs, 
         "question": RunnablePassthrough()}
        | RunnableLambda(safe_generate)
    )
```

### 2. Timeout Configuration

```python
from langchain_openai import ChatOpenAI

# Set reasonable timeouts
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30,  # 30 seconds
    max_retries=2
)

# For retrieval
retriever = vectorstore.as_retriever()

async def retrieve_with_timeout(query, timeout=10):
    try:
        return await asyncio.wait_for(
            retriever.ainvoke(query),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return []
```

## Caching Strategy

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# Development: In-memory
set_llm_cache(InMemoryCache())

# Production: Persistent
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Custom caching for embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store
)
```

## Monitoring & Observability

```python
import logging
import time
from functools import wraps

def monitor_rag(func):
    """Decorator to monitor RAG performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start
            
            logging.info(f"RAG call completed", extra={
                "latency_ms": latency * 1000,
                "status": "success",
                "question_length": len(str(args[0])) if args else 0
            })
            
            return result
            
        except Exception as e:
            logging.error(f"RAG call failed: {e}", extra={
                "status": "error",
                "error_type": type(e).__name__
            })
            raise
    
    return wrapper

@monitor_rag
def ask_rag(question: str) -> str:
    return rag_chain.invoke(question)
```

## Production Checklist

### Before Launch

- [ ] **Chunking optimized** for your content type
- [ ] **Hybrid search** configured
- [ ] **Reranking** tested and tuned
- [ ] **Error handling** for all failure modes
- [ ] **Timeouts** set appropriately
- [ ] **Caching** configured
- [ ] **Monitoring** in place
- [ ] **Evaluation dataset** created
- [ ] **Baseline metrics** established

### Ongoing

- [ ] Monitor **latency** and **error rates**
- [ ] Review **user feedback** regularly
- [ ] Run **automated evaluations** weekly
- [ ] Update **knowledge base** as needed
- [ ] Tune **retrieval parameters** based on metrics

## 🎯 Use Case Modul 5: Company Knowledge Base Bot

```python
#!/usr/bin/env python3
"""
Company Knowledge Base Bot - Use Case Modul 5
Production-ready RAG chatbot untuk dokumen perusahaan.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


class KnowledgeBot:
    """Production-ready RAG Knowledge Base Bot."""
    
    def __init__(self, docs_path: str, persist_dir: str = "./kb_db"):
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.retriever = None
        self.chain = None
    
    def index_documents(self):
        """Load and index documents."""
        print("📂 Loading documents...")
        
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()
        print(f"   Loaded {len(docs)} pages")
        
        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        print(f"   Split into {len(chunks)} chunks")
        
        # Store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"✅ Indexed to {self.persist_dir}")
        
        return chunks
    
    def load_index(self):
        """Load existing index."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        print(f"✅ Loaded index from {self.persist_dir}")
    
    def setup_retriever(self, docs=None):
        """Setup hybrid retriever."""
        semantic = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        if docs:
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = 5
            self.retriever = EnsembleRetriever(
                retrievers=[semantic, bm25],
                weights=[0.6, 0.4]
            )
        else:
            self.retriever = semantic
        
        print("✅ Retriever configured")
    
    def setup_chain(self):
        """Setup RAG chain."""
        prompt = ChatPromptTemplate.from_template("""
        Kamu adalah asisten knowledge base perusahaan.
        Jawab pertanyaan berdasarkan dokumen yang diberikan.
        
        ATURAN:
        1. Jawab HANYA berdasarkan context
        2. Jika tidak ada informasi, katakan "Informasi tidak ditemukan"
        3. Sebutkan sumber jika relevant
        4. Jawab dalam Bahasa Indonesia
        
        Context:
        {context}
        
        Pertanyaan: {question}
        
        Jawaban:
        """)
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                formatted.append(f"[{source}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)
        
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG chain ready")
    
    def ask(self, question: str) -> str:
        """Ask a question."""
        return self.chain.invoke(question)
    
    async def ask_stream(self, question: str):
        """Ask with streaming."""
        async for chunk in self.chain.astream(question):
            yield chunk


def main():
    bot = KnowledgeBot(docs_path="./company_docs")
    
    # Index (first time) or load (subsequent)
    try:
        bot.load_index()
    except:
        chunks = bot.index_documents()
    
    bot.setup_retriever()
    bot.setup_chain()
    
    print("\n🤖 Knowledge Base Bot Ready!")
    print("   Type 'quit' to exit\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue
        
        print("Bot: ", end="", flush=True)
        for chunk in bot.chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
```

---

**Selamat!** 🎉 Kamu sudah menyelesaikan **Modul 5: RAG**!

Kamu sekarang memahami:
- Konsep dan arsitektur RAG
- Document loaders dan text splitters
- Embeddings dan vector stores
- Advanced retrieval techniques
- Evaluasi dan best practices

---

**Selanjutnya:** [Modul 6: Memory & Conversation](/docs/memory/konsep-memory) - Membuat chatbot yang mengingat percakapan.
