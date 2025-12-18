---
sidebar_position: 5
title: Conversational RAG
description: Menggabungkan memory dengan RAG untuk chatbot yang cerdas
---

# Conversational RAG

Menggabungkan **memory** (ingat percakapan) dengan **RAG** (cari dari knowledge base) untuk chatbot yang powerful.

## Masalah: RAG Tanpa History

RAG biasa tidak memahami konteks percakapan.

```python
# Standard RAG
rag_chain.invoke("Apa kebijakan cuti?")
# → "Karyawan mendapat 21 hari cuti per tahun."

rag_chain.invoke("Bagaimana cara mengajukannya?")
# → "Maaf, mengajukan apa? Bisa jelaskan lebih spesifik?"
# ❌ Tidak ingat pertanyaan sebelumnya tentang cuti!
```

## Solusi: History-Aware Retriever

```
┌─────────────────────────────────────────────────────────────┐
│                   Conversational RAG                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Chat History + New Question                                 │
│         │                                                    │
│         ▼                                                    │
│  [Contextualize Question] ──▶ Standalone Question            │
│         │                                                    │
│         ▼                                                    │
│  [Retriever] ──▶ Relevant Documents                         │
│         │                                                    │
│         ▼                                                    │
│  [Generate Answer with Context + History]                   │
│         │                                                    │
│         ▼                                                    │
│       Answer                                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation with LangChain

### 1. Setup Components

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Load and index documents
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### 2. Create History-Aware Retriever

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt untuk contextualize question
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, 
    retriever, 
    contextualize_prompt
)
```

### 3. Create Question-Answer Chain

```python
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah asisten yang membantu menjawab pertanyaan 
berdasarkan dokumen perusahaan. Gunakan context berikut untuk menjawab.
Jika tidak tahu, katakan tidak tahu.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
```

### 4. Create RAG Chain

```python
from langchain.chains import create_retrieval_chain

rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)
```

### 5. Add Memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```

### 6. Use It!

```python
config = {"configurable": {"session_id": "user_123"}}

# First question
response1 = conversational_rag.invoke(
    {"input": "Apa kebijakan cuti di perusahaan?"},
    config=config
)
print(response1["answer"])
# "Menurut handbook, karyawan tetap mendapat 21 hari cuti per tahun..."

# Follow-up - AI understands context!
response2 = conversational_rag.invoke(
    {"input": "Bagaimana cara mengajukannya?"},
    config=config
)
print(response2["answer"])
# "Untuk mengajukan cuti, Anda perlu mengisi form di HRIS
#  minimal 3 hari sebelumnya dan mendapat approval manager..."
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Conversational RAG - Personal Assistant
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()


class ConversationalAssistant:
    def __init__(self, docs_path: str, persist_dir: str = "./conv_rag_db"):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.persist_dir = persist_dir
        self.docs_path = docs_path
        self.store = {}
        self.chain = None
        
    def index_documents(self):
        """Load and index documents."""
        print("📂 Loading documents...")
        
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir
        )
        
        print(f"✅ Indexed {len(chunks)} chunks")
        return vectorstore
    
    def load_vectorstore(self):
        """Load existing vectorstore."""
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
    
    def setup_chain(self, vectorstore):
        """Setup conversational RAG chain."""
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # History-aware retriever
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the chat history and latest question,
reformulate it as a standalone question. Don't answer, just reformulate."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            contextualize_prompt
        )
        
        # QA chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """Kamu adalah asisten perusahaan yang ramah.
Jawab berdasarkan context. Jika tidak tahu, katakan tidak tahu.
Ingat preferensi dan nama user dari percakapan.

Context:
{context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        
        # Add memory
        def get_history(session_id: str):
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]
        
        self.chain = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        print("✅ Chain ready")
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """Chat with the assistant."""
        config = {"configurable": {"session_id": session_id}}
        response = self.chain.invoke({"input": message}, config=config)
        return response["answer"]
    
    def stream_chat(self, message: str, session_id: str = "default"):
        """Stream chat response."""
        config = {"configurable": {"session_id": session_id}}
        
        for chunk in self.chain.stream({"input": message}, config=config):
            if "answer" in chunk:
                yield chunk["answer"]
    
    def get_history(self, session_id: str):
        """Get chat history for a session."""
        if session_id in self.store:
            return self.store[session_id].messages
        return []


def main():
    # Initialize
    assistant = ConversationalAssistant(docs_path="./company_docs")
    
    # Load or create index
    try:
        vectorstore = assistant.load_vectorstore()
        print("✅ Loaded existing index")
    except:
        vectorstore = assistant.index_documents()
    
    assistant.setup_chain(vectorstore)
    
    # Chat loop
    session_id = "demo_session"
    print("\n🤖 Personal Assistant Ready!")
    print("   (type 'quit' to exit, 'history' to see chat history)\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'history':
            history = assistant.get_history(session_id)
            for msg in history:
                role = "You" if msg.type == "human" else "Bot"
                print(f"{role}: {msg.content[:100]}...")
            continue
        
        if not user_input:
            continue
        
        print("Bot: ", end="", flush=True)
        for chunk in assistant.stream_chat(user_input, session_id):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
```

## Streaming with Sources

```python
async def stream_with_sources(question: str, session_id: str):
    config = {"configurable": {"session_id": session_id}}
    
    sources_shown = False
    
    async for chunk in conversational_rag.astream(
        {"input": question},
        config=config
    ):
        # Show sources first (from context)
        if "context" in chunk and not sources_shown:
            print("📚 Sources:")
            for doc in chunk["context"][:2]:
                source = doc.metadata.get("source", "Unknown")
                print(f"  - {source}")
            print("\n💬 Answer: ", end="")
            sources_shown = True
        
        # Stream answer
        if "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
    
    print()
```

## Tips & Best Practices

### 1. Contextualize Prompt Bahasa Indonesia

```python
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Berdasarkan histori chat dan pertanyaan terbaru, 
formulasikan pertanyaan yang bisa dipahami tanpa histori.
JANGAN jawab pertanyaannya, cukup reformulasi jika perlu."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
```

### 2. Limit History Length

```python
def get_limited_history(session_id: str, max_messages: int = 10):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    history = store[session_id]
    
    # Trim if too long
    if len(history.messages) > max_messages:
        history.messages = history.messages[-max_messages:]
    
    return history
```

### 3. Handle No Documents Found

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Jawab berdasarkan context yang diberikan.
Jika context kosong atau tidak relevan, jawab:
"Maaf, saya tidak menemukan informasi tersebut di dokumen."

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
```

## Use Case Modul 6: Personal Assistant

Fitur yang sudah dibuat:
- ✅ Memory - ingat percakapan
- ✅ RAG - cari dari knowledge base
- ✅ History-aware - paham konteks
- ✅ Streaming - real-time response
- ✅ Multi-session - support banyak user

## Ringkasan

1. **Conversational RAG** = Memory + RAG
2. **History-aware retriever** reformulates questions
3. `create_history_aware_retriever()` + `create_retrieval_chain()`
4. Wrap dengan **RunnableWithMessageHistory**
5. Perfect untuk **chatbot dengan knowledge base**

---

**Selamat!** 🎉 Kamu sudah menyelesaikan **Modul 6: Memory & Conversation**!

Kamu sekarang bisa membuat:
- Chatbot yang mengingat percakapan
- Berbagai memory strategies
- Conversational RAG systems

---

**Selanjutnya:** [Modul 7: Agents & Tool Calling](/docs/agents/konsep-agents) - Membuat LLM yang bisa mengambil aksi.
