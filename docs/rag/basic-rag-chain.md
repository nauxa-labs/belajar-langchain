---
sidebar_position: 7
title: Basic RAG Chain
description: Membangun pipeline RAG lengkap dengan LCEL
---

# Basic RAG Chain

Setelah memahami komponen-komponen RAG, saatnya menggabungkan semuanya menjadi **pipeline yang berfungsi**.

## End-to-End RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Documents ──▶ Split ──▶ Embed ──▶ Store                        │
│                                      │                           │
│                                      ▼                           │
│  Query ──▶ Embed ──▶ Retrieve ──▶ Combine ──▶ LLM ──▶ Answer    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Indexing Pipeline

Memproses dokumen dan menyimpan ke vector store.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load documents
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")

# 3. Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Indexed {len(chunks)} chunks to vector store")
```

## Step 2: Retrieval Chain

Query dan retrieve relevant documents.

```python
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Test retrieval
query = "Berapa hari cuti per tahun?"
relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(doc.page_content[:200])
```

## Step 3: Generation Chain

Combine context + query → LLM → Answer.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# RAG prompt template
rag_prompt = ChatPromptTemplate.from_template("""
Jawab pertanyaan berikut berdasarkan context yang diberikan.
Jika informasi tidak ada di context, katakan "Saya tidak menemukan informasi tersebut."

Context:
{context}

Pertanyaan: {question}

Jawaban:
""")

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Format documents to string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

## Complete RAG Chain with LCEL

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Method 1: Simple chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Use
answer = rag_chain.invoke("Berapa hari cuti yang didapat karyawan?")
print(answer)
```

### Method 2: With Source Documents

```python
from langchain_core.runnables import RunnableParallel

# Chain that also returns source documents
rag_chain_with_sources = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(
    answer=lambda x: (
        rag_prompt 
        | llm 
        | StrOutputParser()
    ).invoke({
        "context": format_docs(x["context"]),
        "question": x["question"]
    })
)

result = rag_chain_with_sources.invoke("Apa kebijakan work from home?")

print("Answer:", result["answer"])
print("\nSources:")
for doc in result["context"]:
    print(f"- {doc.metadata.get('source')}")
```

## Using create_retrieval_chain (High-level API)

LangChain provides a simpler high-level function:

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create document chain
document_chain = create_stuff_documents_chain(llm, rag_prompt)

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Use
result = retrieval_chain.invoke({"input": "Berapa hari cuti per tahun?"})

print(result["answer"])
print(result["context"])  # Source documents
```

## Streaming RAG

Real-time streaming untuk better UX.

```python
async def stream_rag_answer(question: str):
    print(f"Question: {question}\n")
    print("Answer: ", end="")
    
    async for chunk in rag_chain.astream(question):
        print(chunk, end="", flush=True)
    
    print()

# Run
import asyncio
asyncio.run(stream_rag_answer("Jelaskan kebijakan cuti tahunan"))
```

### Streaming with Sources

```python
from langchain_core.runnables import RunnableParallel

async def stream_with_sources(question: str):
    # First, get sources
    docs = await retriever.ainvoke(question)
    
    print("Sources:")
    for doc in docs:
        print(f"- {doc.metadata.get('source', 'Unknown')}")
    
    print("\nAnswer: ", end="")
    
    # Then stream answer
    chain = rag_prompt | llm | StrOutputParser()
    async for chunk in chain.astream({
        "context": format_docs(docs),
        "question": question
    }):
        print(chunk, end="", flush=True)
    
    print()
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Basic RAG Chain - Complete Example
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- Indexing ---
def create_vectorstore(pdf_path: str, persist_dir: str = "./chroma_db"):
    """Create or load vector store from PDF."""
    
    # Check if already exists
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings()
        )
        if vectorstore._collection.count() > 0:
            print(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
            return vectorstore
    except:
        pass
    
    # Create new
    print(f"Creating new vectorstore from {pdf_path}...")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
    
    print(f"Created vectorstore with {len(chunks)} chunks")
    return vectorstore

# --- RAG Chain ---
def create_rag_chain(vectorstore):
    """Create RAG chain."""
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
    Kamu adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen.
    Jawab dengan jelas dan ringkas berdasarkan context berikut.
    Jika tidak ada informasi di context, katakan "Informasi tidak ditemukan dalam dokumen."
    
    Context:
    {context}
    
    Pertanyaan: {question}
    
    Jawaban:
    """)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# --- Main ---
def main():
    # Create vectorstore
    vectorstore = create_vectorstore("handbook.pdf")
    
    # Create chain
    chain = create_rag_chain(vectorstore)
    
    # Interactive loop
    print("\n🤖 RAG Chatbot Ready! (type 'quit' to exit)\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ('quit', 'exit', 'q'):
            break
        
        if not question:
            continue
        
        print("Bot: ", end="")
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
```

## Error Handling

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

def safe_retrieve(query: str):
    try:
        docs = retriever.invoke(query)
        if not docs:
            return [Document(page_content="No relevant documents found.")]
        return docs
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [Document(page_content="Error retrieving documents.")]

safe_chain = (
    {"context": RunnableLambda(safe_retrieve) | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

## Ringkasan

1. **Indexing**: Load → Split → Embed → Store
2. **Retrieval**: Query → Retrieve relevant chunks
3. **Generation**: Context + Question → LLM → Answer
4. **LCEL** membuat pipeline clean dan composable
5. **Streaming** untuk responsive UX
6. **create_retrieval_chain** untuk high-level API

---

**Selanjutnya:** [Advanced Retrieval](/docs/rag/advanced-retrieval) - Teknik retrieval lanjutan.
