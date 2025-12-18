---
sidebar_position: 2
title: Message History
description: Menyimpan dan mengelola conversation history
---

# Message History

Message History adalah komponen untuk **menyimpan percakapan**. LangChain menyediakan berbagai backend storage.

## ChatMessageHistory (In-Memory)

Paling simple - data hilang saat aplikasi restart.

```python
from langchain_community.chat_message_histories import ChatMessageHistory

# Create history
history = ChatMessageHistory()

# Add messages
history.add_user_message("Halo, siapa namamu?")
history.add_ai_message("Halo! Saya adalah AI assistant. Ada yang bisa saya bantu?")

history.add_user_message("Ceritakan tentang Python")
history.add_ai_message("Python adalah bahasa pemrograman yang populer...")

# Access messages
print(history.messages)
# [HumanMessage(...), AIMessage(...), HumanMessage(...), AIMessage(...)]

# Clear history
history.clear()
```

## Session-based Storage

Untuk multi-user, kita perlu memisahkan history per session.

```python
from langchain_community.chat_message_histories import ChatMessageHistory

# Store untuk semua sessions
session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create history for a session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Usage
user1_history = get_session_history("user_001")
user1_history.add_user_message("Halo dari user 1")

user2_history = get_session_history("user_002")
user2_history.add_user_message("Halo dari user 2")

# Histories are separate
print(len(user1_history.messages))  # 1
print(len(user2_history.messages))  # 1
```

## Redis (Production)

Persistent storage dengan Redis - data survive restart.

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379"
    )

# Usage
history = get_redis_history("user_123")
history.add_user_message("Halo!")
history.add_ai_message("Hai! Apa kabar?")

# Messages persist in Redis
# Key: "message_store:user_123"
```

### Installation

```bash
pip install redis
```

### Redis Configuration

```python
# With password
history = RedisChatMessageHistory(
    session_id="user_123",
    url="redis://:password@localhost:6379/0"
)

# With TTL (auto-expire)
history = RedisChatMessageHistory(
    session_id="user_123",
    url="redis://localhost:6379",
    ttl=3600  # Expire after 1 hour
)
```

## SQLite (Local Persistent)

Good untuk development dan aplikasi single-server.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sql_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )

# Usage
history = get_sql_history("user_456")
history.add_user_message("Simpan di database!")
```

### Installation

```bash
pip install sqlalchemy
```

## PostgreSQL (Production)

Untuk production dengan existing Postgres infrastructure.

```python
from langchain_community.chat_message_histories import PostgresChatMessageHistory

def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string="postgresql://user:pass@localhost:5432/chatdb"
    )
```

## MongoDB

Document-based storage.

```python
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

def get_mongo_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://localhost:27017",
        database_name="chat_db",
        collection_name="message_histories"
    )
```

```bash
pip install langchain-mongodb
```

## File-based Storage

Simple JSON file storage.

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

def get_file_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(
        file_path=f"./histories/{session_id}.json"
    )

# Creates JSON file:
# {"messages": [{"type": "human", "content": "..."}, ...]}
```

## Custom Message History

Buat sendiri untuk storage khusus.

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from typing import List
import json

class CustomFileHistory(BaseChatMessageHistory):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._messages: List[BaseMessage] = []
        self._load()
    
    def _load(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self._messages = messages_from_dict(data)
        except FileNotFoundError:
            self._messages = []
    
    def _save(self):
        with open(self.file_path, 'w') as f:
            json.dump(messages_to_dict(self._messages), f)
    
    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._save()
    
    def clear(self) -> None:
        self._messages = []
        self._save()
```

## Managing Message Count

### Window-based Trimming

```python
from langchain_core.messages import trim_messages

def get_windowed_history(session_id: str, max_messages: int = 20):
    history = get_session_history(session_id)
    
    # Trim to last N messages
    if len(history.messages) > max_messages:
        trimmed = trim_messages(
            history.messages,
            max_tokens=None,
            token_counter=len,  # Simple count
            strategy="last",
            include_system=True
        )
        
        # Replace with trimmed
        history.clear()
        for msg in trimmed:
            history.add_message(msg)
    
    return history
```

### Token-based Trimming

```python
import tiktoken

def count_tokens(messages, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return sum(len(encoding.encode(m.content)) for m in messages)

def get_token_limited_history(session_id: str, max_tokens: int = 4000):
    history = get_session_history(session_id)
    messages = history.messages
    
    while count_tokens(messages) > max_tokens and len(messages) > 2:
        messages = messages[2:]  # Remove oldest pair
    
    return messages
```

## Comparison Table

| Storage | Persistent | Multi-process | Best For |
|---------|------------|---------------|----------|
| ChatMessageHistory | ❌ | ❌ | Testing, demos |
| Redis | ✅ | ✅ | Production, scale |
| SQLite | ✅ | ❌ | Development, single server |
| PostgreSQL | ✅ | ✅ | Production with existing PG |
| MongoDB | ✅ | ✅ | Document-based apps |
| File | ✅ | ❌ | Simple persistence |

## Best Practices

### 1. Use Session IDs

```python
from uuid import uuid4

# Generate on first visit
session_id = str(uuid4())

# Or derive from user
session_id = f"user_{user_id}_chat_{chat_id}"
```

### 2. Set TTL for Cleanup

```python
# Redis with 24 hour TTL
history = RedisChatMessageHistory(
    session_id=session_id,
    url="redis://localhost:6379",
    ttl=86400  # 24 hours
)
```

### 3. Handle Errors Gracefully

```python
def safe_get_history(session_id: str):
    try:
        return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    except Exception as e:
        print(f"Redis error: {e}, falling back to in-memory")
        return ChatMessageHistory()
```

## Ringkasan

1. **ChatMessageHistory** - in-memory, untuk testing
2. **RedisChatMessageHistory** - production, scalable
3. **SQLChatMessageHistory** - local persistent
4. **Session ID** wajib untuk multi-user
5. **TTL** untuk auto-cleanup
6. Pilih berdasarkan **infrastructure** yang sudah ada

---

**Selanjutnya:** [RunnableWithMessageHistory](/docs/memory/runnable-with-history) - Integrasi memory ke LCEL chains.
