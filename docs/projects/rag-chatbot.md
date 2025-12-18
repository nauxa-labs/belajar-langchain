---
sidebar_position: 2
title: "Proyek 1: RAG Chatbot"
description: Chatbot dengan knowledge base dan memory
---

# Proyek 1: RAG Chatbot untuk Dokumentasi

Membangun chatbot yang menjawab pertanyaan berdasarkan **custom documentation** dengan kemampuan **mengingat percakapan**.

## Requirements

### Fitur Utama
- ✅ Load dokumentasi dari multiple sources
- ✅ Hybrid retrieval (semantic + keyword)
- ✅ Conversational memory
- ✅ Source attribution
- ✅ Streaming responses

### Tech Stack
- LangChain + LCEL
- OpenAI (GPT-4o-mini + Embeddings)
- ChromaDB (vector store)
- FastAPI + LangServe

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Chatbot System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Question                                              │
│        │                                                     │
│        ▼                                                     │
│   [Context from History]                                     │
│        │                                                     │
│        ▼                                                     │
│   [Reformulate to Standalone Question]                      │
│        │                                                     │
│        ▼                                                     │
│   [Hybrid Retrieval]                                        │
│        │ Semantic + BM25                                    │
│        ▼                                                     │
│   [Rerank Documents]                                        │
│        │                                                     │
│        ▼                                                     │
│   [Generate Answer with Sources]                            │
│        │                                                     │
│        ▼                                                     │
│   Streaming Response + Citations                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Document Loading

```python
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)

def load_documents(docs_path: str):
    """Load documents from various sources."""
    
    loaders = [
        DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
    ]
    
    all_docs = []
    for loader in loaders:
        try:
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Warning: {e}")
    
    return all_docs
```

### Step 2: Text Splitting & Indexing

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_index(documents, persist_dir: str = "./chroma_db"):
    """Split and index documents."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
    
    return vectorstore
```

### Step 3: Hybrid Retrieval

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def create_hybrid_retriever(vectorstore, documents, k: int = 4):
    """Create hybrid semantic + keyword retriever."""
    
    # Semantic retriever
    semantic = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # BM25 keyword retriever
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    
    # Combine with equal weights
    hybrid = EnsembleRetriever(
        retrievers=[semantic, bm25],
        weights=[0.5, 0.5]
    )
    
    return hybrid
```

### Step 4: Conversational RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_conversational_rag(retriever):
    """Create conversational RAG chain."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    
    # 1. Contextualize question
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and latest question,
reformulate it as a standalone question. Don't answer."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    
    # 2. QA with sources
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Kamu adalah asisten yang menjawab berdasarkan dokumentasi.

Aturan:
1. Jawab HANYA berdasarkan context yang diberikan
2. Jika tidak ada di context, katakan "Maaf, informasi tidak ditemukan"
3. Sertakan sumber di akhir jawaban

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Full chain
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    return rag_chain
```

### Step 5: Memory Integration

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_chatbot(rag_chain):
    """Wrap with memory."""
    
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
```

### Step 6: FastAPI Server

```python
from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel

app = FastAPI(title="Documentation Chatbot")

class ChatInput(BaseModel):
    input: str
    session_id: str = "default"

# Initialize
docs = load_documents("./documentation")
vectorstore = create_index(docs)
retriever = create_hybrid_retriever(vectorstore, docs)
rag_chain = create_conversational_rag(retriever)
chatbot = create_chatbot(rag_chain)

add_routes(app, chatbot, path="/chat")

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

## Complete Code

```python
#!/usr/bin/env python3
"""
RAG Chatbot untuk Dokumentasi
Proyek 1 - Modul 10
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()


class DocumentationChatbot:
    def __init__(self, docs_path: str, persist_dir: str = "./chroma_db"):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        self.embeddings = OpenAIEmbeddings()
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        self.store = {}
        
        self._setup()
    
    def _setup(self):
        # Load and index
        loader = DirectoryLoader(self.docs_path, glob="**/*.*", loader_cls=TextLoader)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        self.vectorstore = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_dir
        )
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create chain
        contextualize = ChatPromptTemplate.from_messages([
            ("system", "Reformulate the question as standalone. Don't answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize
        )
        
        qa = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context. Cite sources.\n\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        doc_chain = create_stuff_documents_chain(self.llm, qa)
        rag_chain = create_retrieval_chain(history_retriever, doc_chain)
        
        self.chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    
    def _get_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def chat(self, message: str, session_id: str = "default") -> str:
        config = {"configurable": {"session_id": session_id}}
        result = self.chain.invoke({"input": message}, config=config)
        return result["answer"]
    
    async def stream(self, message: str, session_id: str = "default"):
        config = {"configurable": {"session_id": session_id}}
        async for chunk in self.chain.astream({"input": message}, config=config):
            if "answer" in chunk:
                yield chunk["answer"]


def main():
    chatbot = DocumentationChatbot("./docs")
    
    print("🤖 Documentation Chatbot Ready!")
    print("Type 'quit' to exit, 'clear' to reset history\n")
    
    session_id = "cli-session"
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ('quit', 'exit'):
            break
        
        if user_input.lower() == 'clear':
            chatbot.store.pop(session_id, None)
            print("History cleared.\n")
            continue
        
        if not user_input:
            continue
        
        response = chatbot.chat(user_input, session_id)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
```

## Testing

```python
# tests/test_chatbot.py
import pytest

def test_basic_qa():
    chatbot = DocumentationChatbot("./test_docs")
    response = chatbot.chat("What is Python?")
    assert len(response) > 0

def test_memory():
    chatbot = DocumentationChatbot("./test_docs")
    chatbot.chat("My name is Alice", "test-session")
    response = chatbot.chat("What is my name?", "test-session")
    assert "Alice" in response

def test_no_info():
    chatbot = DocumentationChatbot("./test_docs")
    response = chatbot.chat("What is the meaning of life?")
    assert "tidak ditemukan" in response.lower() or "don't know" in response.lower()
```

## Deployment

```bash
# Run locally
uvicorn server:app --reload

# Docker
docker build -t docs-chatbot .
docker run -p 8000:8000 docs-chatbot
```

## Improvements

Ide untuk mengembangkan lebih lanjut:
- [ ] Add reranking dengan Cohere
- [ ] Implement caching
- [ ] Add feedback collection
- [ ] Multi-language support
- [ ] Admin panel untuk manage docs

---

**Selanjutnya:** [Proyek 2: Research Agent](/docs/projects/research-agent)
