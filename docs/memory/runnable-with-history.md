---
sidebar_position: 3
title: RunnableWithMessageHistory
description: Cara modern menambahkan memory ke LCEL chains
---

# RunnableWithMessageHistory

`RunnableWithMessageHistory` adalah cara modern dan recommended untuk menambahkan memory ke chains di LangChain.

## Konsep Dasar

```
┌─────────────────────────────────────────────────────────────┐
│              RunnableWithMessageHistory                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input ──▶ [Load History] ──▶ [Chain] ──▶ [Save to History] │
│                 │                              │             │
│                 └──────── Message Store ───────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Basic Setup

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Create base chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah asisten yang ramah dan membantu."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | llm | StrOutputParser()

# 2. Setup message store
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 3. Wrap with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

## Using the Chain

```python
# Conversation with session_id
config = {"configurable": {"session_id": "user_123"}}

# First message
response1 = chain_with_history.invoke(
    {"input": "Halo! Nama saya Budi."},
    config=config
)
print(response1)
# "Halo Budi! Senang berkenalan. Ada yang bisa saya bantu?"

# Second message - AI remembers!
response2 = chain_with_history.invoke(
    {"input": "Apa nama saya?"},
    config=config
)
print(response2)
# "Nama Anda adalah Budi."

# Different session - different conversation
config2 = {"configurable": {"session_id": "user_456"}}
response3 = chain_with_history.invoke(
    {"input": "Apa nama saya?"},
    config=config2
)
print(response3)
# "Maaf, saya tidak tahu nama Anda. Boleh perkenalkan diri?"
```

## Key Parameters

### input_messages_key

Key dalam input dict yang berisi pesan user.

```python
# Jika input Anda: {"question": "Halo"}
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",  # Match your input key
    history_messages_key="history"
)
```

### history_messages_key

Key dalam prompt yang menjadi placeholder untuk history.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="chat_history"),  # This name
    ("human", "{input}")
])

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"  # Must match
)
```

### output_messages_key

Jika chain return dict, specify key untuk AI response.

```python
# Chain yang return dict
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="response"  # If chain returns {"response": "..."}
)
```

## With Persistent Storage

### Redis Example

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=3600  # 1 hour
    )

chain_with_redis = RunnableWithMessageHistory(
    chain,
    get_redis_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### SQLite Example

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sql_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat.db"
    )

chain_with_sql = RunnableWithMessageHistory(
    chain,
    get_sql_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

## Streaming with History

```python
async def stream_with_memory(question: str, session_id: str):
    config = {"configurable": {"session_id": session_id}}
    
    async for chunk in chain_with_history.astream(
        {"input": question},
        config=config
    ):
        print(chunk, end="", flush=True)
    print()

# Usage
import asyncio
asyncio.run(stream_with_memory("Ceritakan tentang Python", "user_123"))
```

## Complete Chatbot Example

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# Chain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah asisten AI yang ramah bernama Aria.
Kamu selalu menjawab dalam Bahasa Indonesia.
Ingat nama dan preferensi user dari percakapan."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = prompt | llm | StrOutputParser()

# Store
store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chain with memory
chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

def chat(session_id: str):
    config = {"configurable": {"session_id": session_id}}
    
    print("🤖 Aria: Halo! Saya Aria, asisten AI Anda. (ketik 'quit' untuk keluar)\n")
    
    while True:
        user_input = input("Anda: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("🤖 Aria: Sampai jumpa!")
            break
        
        if not user_input:
            continue
        
        print("🤖 Aria: ", end="")
        for chunk in chatbot.stream({"input": user_input}, config=config):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    chat("session_001")
```

## Multiple Configurable Fields

Support multiple session parameters.

```python
from langchain_core.runnables import ConfigurableFieldSpec

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique user identifier",
            default="",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique conversation identifier",
            default="",
            is_shared=True
        )
    ]
)

# Custom history function
def get_history_multi(user_id: str, conversation_id: str):
    session_key = f"{user_id}:{conversation_id}"
    if session_key not in store:
        store[session_key] = ChatMessageHistory()
    return store[session_key]

# Usage
config = {
    "configurable": {
        "user_id": "user_123",
        "conversation_id": "conv_456"
    }
}
response = chain_with_history.invoke({"input": "Hello"}, config=config)
```

## API Server Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"session_id": request.session_id}}
    
    response = await chatbot.ainvoke(
        {"input": request.message},
        config=config
    )
    
    return ChatResponse(response=response)

# GET history
@app.get("/history/{session_id}")
async def get_history_endpoint(session_id: str):
    history = get_history(session_id)
    return {
        "messages": [
            {"role": "human" if hasattr(m, 'type') and m.type == 'human' else "ai", 
             "content": m.content}
            for m in history.messages
        ]
    }
```

## Error Handling

```python
from langchain_core.runnables import RunnableLambda

def safe_get_history(session_id: str):
    try:
        return get_redis_history(session_id)
    except Exception as e:
        print(f"History error: {e}")
        return ChatMessageHistory()  # Fallback

chain_with_history = RunnableWithMessageHistory(
    chain,
    safe_get_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

## Ringkasan

1. **RunnableWithMessageHistory** - cara modern untuk memory
2. **get_session_history** function - return history store
3. **input_messages_key** - key untuk input user
4. **history_messages_key** - match dengan MessagesPlaceholder
5. **config** dengan session_id untuk multi-user
6. Support **streaming** dan **async**

---

**Selanjutnya:** [Jenis Memory](/docs/memory/memory-types) - Buffer, Window, Summary, dan Vector memory.
