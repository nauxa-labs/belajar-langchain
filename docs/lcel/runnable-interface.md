---
sidebar_position: 2
title: Runnable Interface
description: Method utama yang tersedia di setiap Runnable - invoke, stream, batch, dan async variants
---

# Runnable Interface

Semua komponen LCEL mengimplementasikan **Runnable interface**. Ini adalah kontrak yang memastikan setiap komponen memiliki methods yang sama untuk eksekusi.

## Core Methods

Setiap Runnable memiliki 6 methods utama:

| Method | Type | Use Case |
|--------|------|----------|
| `invoke()` | Sync | Single input, tunggu hasil |
| `stream()` | Sync | Single input, stream output |
| `batch()` | Sync | Multiple inputs sekaligus |
| `ainvoke()` | Async | Single input dalam async context |
| `astream()` | Async | Stream dalam async context |
| `abatch()` | Async | Batch dalam async context |

## `invoke()` - Single Execution

Method paling dasar untuk menjalankan chain dengan satu input.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Jelaskan {topik} dalam 2 kalimat")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Single invoke
result = chain.invoke({"topik": "machine learning"})
print(result)
```

### Return Value

`invoke()` mengembalikan output langsung:

```python
result = chain.invoke({"topik": "AI"})
print(type(result))  # <class 'str'>
print(result)        # "AI adalah..."
```

## `stream()` - Token Streaming

Menghasilkan output secara bertahap - ideal untuk real-time display.

```python
# Stream output token by token
for chunk in chain.stream({"topik": "quantum computing"}):
    print(chunk, end="", flush=True)
print()  # New line at end
```

Output (muncul bertahap):

```
Quantum computing adalah...paradigma komputasi...yang menggunakan...
```

### Streaming dengan Progress

```python
import sys

def stream_with_progress(chain, input_data):
    """Stream with character count."""
    total_chars = 0
    
    for chunk in chain.stream(input_data):
        print(chunk, end="", flush=True)
        total_chars += len(chunk)
    
    print(f"\n\n[Total: {total_chars} characters]")

stream_with_progress(chain, {"topik": "neural networks"})
```

## `batch()` - Parallel Processing

Memproses multiple inputs secara bersamaan.

```python
# Batch processing
topics = [
    {"topik": "Python"},
    {"topik": "JavaScript"},
    {"topik": "Rust"}
]

results = chain.batch(topics)

for topic, result in zip(topics, results):
    print(f"=== {topic['topik']} ===")
    print(result)
    print()
```

### Batch dengan Max Concurrency

Kontrol berapa banyak requests paralel:

```python
# Limit to 2 concurrent requests
results = chain.batch(
    topics,
    config={"max_concurrency": 2}
)
```

### Batch dengan Return Exceptions

Jangan stop saat error, kumpulkan semua hasil:

```python
results = chain.batch(
    topics,
    return_exceptions=True  # Return errors instead of raising
)

for topic, result in zip(topics, results):
    if isinstance(result, Exception):
        print(f"Error for {topic}: {result}")
    else:
        print(f"Success for {topic}")
```

## Async Methods

Untuk aplikasi async seperti FastAPI, gunakan async variants.

### `ainvoke()`

```python
import asyncio

async def main():
    result = await chain.ainvoke({"topik": "blockchain"})
    print(result)

asyncio.run(main())
```

### `astream()`

```python
async def stream_async():
    async for chunk in chain.astream({"topik": "AI ethics"}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_async())
```

### `abatch()`

```python
async def batch_async():
    topics = [
        {"topik": "Docker"},
        {"topik": "Kubernetes"},
        {"topik": "Terraform"}
    ]
    
    results = await chain.abatch(topics)
    
    for topic, result in zip(topics, results):
        print(f"{topic['topik']}: {result[:50]}...")

asyncio.run(batch_async())
```

## Advanced: `astream_events()`

Untuk mendapatkan events detail dari setiap step dalam chain.

```python
async def stream_events():
    async for event in chain.astream_events(
        {"topik": "GPT"},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_start":
            print(f"🚀 Model started: {event['name']}")
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            print(content, end="", flush=True)
        elif kind == "on_chat_model_end":
            print(f"\n✅ Model finished")

asyncio.run(stream_events())
```

Output:

```
🚀 Model started: ChatOpenAI
GPT (Generative Pre-trained Transformer) adalah...
✅ Model finished
```

### Event Types

| Event | Trigger |
|-------|---------|
| `on_chain_start` | Chain mulai dieksekusi |
| `on_chain_end` | Chain selesai |
| `on_chat_model_start` | LLM mulai generate |
| `on_chat_model_stream` | Setiap token di-stream |
| `on_chat_model_end` | LLM selesai |
| `on_parser_start` | Parser mulai |
| `on_parser_end` | Parser selesai |

## Config Parameter

Semua methods menerima `config` parameter untuk customization:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_concurrency=5,
    run_name="my-translation-chain",
    tags=["production", "translation"],
    metadata={"user_id": "123", "session": "abc"}
)

result = chain.invoke({"topik": "AI"}, config=config)
```

### Common Config Options

| Option | Description |
|--------|-------------|
| `max_concurrency` | Max parallel executions |
| `run_name` | Name for tracing |
| `tags` | Tags for filtering traces |
| `metadata` | Custom metadata |
| `callbacks` | Callback handlers |
| `recursion_limit` | Max recursion depth |

## Callbacks

Monitor chain execution dengan callbacks:

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🚀 LLM Starting with {len(prompts)} prompts")
    
    def on_llm_end(self, response, **kwargs):
        print(f"✅ LLM Finished")
    
    def on_chain_error(self, error, **kwargs):
        print(f"❌ Error: {error}")

result = chain.invoke(
    {"topik": "callbacks"},
    config={"callbacks": [MyHandler()]}
)
```

## Practical Example: API dengan Streaming

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

app = FastAPI()

chain = (
    ChatPromptTemplate.from_template("Write about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

@app.get("/generate")
async def generate(topic: str):
    async def generate_stream():
        async for chunk in chain.astream({"topic": topic}):
            yield chunk
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain"
    )

@app.get("/generate-sync")
def generate_sync(topic: str):
    return chain.invoke({"topic": topic})
```

## Method Comparison

| Method | Blocking? | Output | Best For |
|--------|-----------|--------|----------|
| `invoke()` | Yes | Single value | Simple scripts |
| `stream()` | Yes (generator) | Iterator | Real-time display |
| `batch()` | Yes | List | Bulk processing |
| `ainvoke()` | No | Awaitable | Web servers |
| `astream()` | No | AsyncIterator | Async streaming |
| `abatch()` | No | Awaitable[List] | Async bulk |

## Performance Tips

### 1. Gunakan Batch untuk Multiple Items

```python
# ❌ Slow - sequential
results = [chain.invoke(item) for item in items]

# ✅ Fast - parallel
results = chain.batch(items)
```

### 2. Gunakan Async di Web Servers

```python
# ❌ Blocks event loop
@app.get("/")
def handler():
    return chain.invoke(input)

# ✅ Non-blocking
@app.get("/")
async def handler():
    return await chain.ainvoke(input)
```

### 3. Stream untuk Long Responses

```python
# ❌ User waits until complete
result = chain.invoke(input)

# ✅ User sees progress
for chunk in chain.stream(input):
    yield chunk
```

## Ringkasan

1. **6 core methods**: invoke, stream, batch + async variants
2. **`invoke()`** - single execution, wait for result
3. **`stream()`** - token-by-token output
4. **`batch()`** - parallel processing multiple inputs
5. **Async variants** (a-prefix) untuk non-blocking execution
6. **`astream_events()`** untuk detailed event monitoring
7. Gunakan **config** untuk customization dan tracing

---

**Selanjutnya:** [Composing Runnables](/docs/lcel/composing-runnables) - Cara menggabungkan multiple runnables menjadi chain kompleks.
