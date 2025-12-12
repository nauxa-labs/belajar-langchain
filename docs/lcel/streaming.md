---
sidebar_position: 6
title: Streaming
description: Streaming tokens dan events untuk responsive user experience
---

# Streaming Deep Dive

Streaming adalah fitur penting untuk UX yang baik. Daripada menunggu seluruh response, user bisa melihat progress secara real-time.

## Mengapa Streaming?

| Tanpa Streaming | Dengan Streaming |
|-----------------|------------------|
| User menunggu 5-10 detik | Output muncul dalam ~500ms pertama |
| Tidak ada progress indication | User melihat AI "berpikir" |
| Poor UX untuk long responses | Engaging experience |

## Basic Streaming

### `stream()` Method

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Write a story about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Stream output
for chunk in chain.stream({"topic": "a robot learning to cook"}):
    print(chunk, end="", flush=True)
print()  # Newline at end
```

### Chunk Structure

Setiap chunk adalah bagian dari output:

```python
for i, chunk in enumerate(chain.stream({"topic": "AI"})):
    print(f"Chunk {i}: '{chunk}'")
```

Output:

```
Chunk 0: 'Artificial'
Chunk 1: ' Intelligence'
Chunk 2: ' ('
Chunk 3: 'AI'
Chunk 4: ')'
Chunk 5: ' is'
...
```

## Async Streaming

Untuk web applications.

```python
import asyncio

async def stream_story():
    async for chunk in chain.astream({"topic": "space exploration"}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_story())
```

### FastAPI Streaming Response

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_response(topic: str):
    async def generate():
        async for chunk in chain.astream({"topic": topic}):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
```

## Server-Sent Events (SSE)

Standard untuk streaming ke browser.

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import json

app = FastAPI()

@app.get("/sse")
async def sse_endpoint(topic: str):
    async def event_generator():
        async for chunk in chain.astream({"topic": topic}):
            yield {
                "event": "message",
                "data": json.dumps({"content": chunk})
            }
        yield {
            "event": "done",
            "data": json.dumps({"status": "complete"})
        }
    
    return EventSourceResponse(event_generator())
```

### Frontend SSE Client

```javascript
const eventSource = new EventSource('/sse?topic=AI');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    document.getElementById('output').textContent += data.content;
};

eventSource.addEventListener('done', () => {
    eventSource.close();
    console.log('Stream complete');
});
```

## `astream_events()` - Detailed Events

Mendapatkan events detail dari setiap komponen dalam chain.

```python
async def stream_with_events():
    async for event in chain.astream_events(
        {"topic": "machine learning"},
        version="v2"
    ):
        kind = event["event"]
        name = event.get("name", "")
        
        if kind == "on_chain_start":
            print(f"🚀 Chain started: {name}")
        
        elif kind == "on_chat_model_start":
            print(f"💭 LLM thinking...")
        
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)
        
        elif kind == "on_chat_model_end":
            print(f"\n✅ LLM finished")
        
        elif kind == "on_chain_end":
            print(f"🏁 Chain complete")

asyncio.run(stream_with_events())
```

### Event Types

| Event | When | Data |
|-------|------|------|
| `on_chain_start` | Chain begins | Input data |
| `on_chain_stream` | Chain outputs | Chunk |
| `on_chain_end` | Chain completes | Final output |
| `on_chat_model_start` | LLM starts | Messages |
| `on_chat_model_stream` | Token generated | Token chunk |
| `on_chat_model_end` | LLM finishes | Full response |
| `on_parser_start` | Parser begins | Input |
| `on_parser_end` | Parser finishes | Parsed output |

## Filtering Events

Hanya process events yang diperlukan.

```python
async def stream_llm_only():
    async for event in chain.astream_events(
        {"topic": "Python"},
        version="v2",
        include_names=["ChatOpenAI"]  # Only LLM events
    ):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="")

# Or exclude specific events
async for event in chain.astream_events(
    input_data,
    version="v2",
    exclude_names=["StrOutputParser"]  # Skip parser events
):
    ...
```

## Streaming dengan Callbacks

Alternative untuk fine-grained control.

```python
from langchain_core.callbacks import BaseCallbackHandler

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
    
    def on_llm_start(self, *args, **kwargs):
        print("🤖 ", end="")
    
    def on_llm_end(self, *args, **kwargs):
        print("\n✅ Done")

# Use with streaming=True on LLM
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | llm | parser

result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [StreamingHandler()]}
)
```

## Collecting Streamed Output

Kadang kita perlu stream DAN collect full output.

```python
async def stream_and_collect():
    collected = []
    
    async for chunk in chain.astream({"topic": "blockchain"}):
        print(chunk, end="", flush=True)
        collected.append(chunk)
    
    full_response = "".join(collected)
    print(f"\n\nFull response ({len(full_response)} chars)")
    return full_response

result = asyncio.run(stream_and_collect())
```

## Streaming JSON/Structured Output

### Partial JSON Streaming

```python
from langchain_core.output_parsers import JsonOutputParser

chain = (
    ChatPromptTemplate.from_template("Return JSON about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | JsonOutputParser()
)

# JSON dibangun secara incremental
async for partial in chain.astream({"topic": "Python"}):
    print(f"Partial: {partial}")
```

Output:

```
Partial: {}
Partial: {'name': 'Python'}
Partial: {'name': 'Python', 'type': 'programming'}
Partial: {'name': 'Python', 'type': 'programming language'}
...
```

## Streaming dengan Progress Bar

```python
from tqdm import tqdm
import sys

def stream_with_progress(chain, input_data, estimated_tokens=500):
    """Stream with progress bar."""
    pbar = tqdm(total=estimated_tokens, desc="Generating", unit="tokens")
    
    collected = []
    for chunk in chain.stream(input_data):
        collected.append(chunk)
        pbar.update(1)
        sys.stdout.write(f"\r{chunk}")
        sys.stdout.flush()
    
    pbar.close()
    return "".join(collected)
```

## Practical Example: Chat UI

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chat chain
chat_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

async def chat_with_streaming(user_input: str, history: list):
    """Chat with streaming response."""
    print("Assistant: ", end="")
    
    collected = []
    async for chunk in chat_chain.astream({
        "input": user_input,
        "history": history
    }):
        print(chunk, end="", flush=True)
        collected.append(chunk)
    
    print()  # Newline
    full_response = "".join(collected)
    
    # Update history
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=full_response))
    
    return full_response

# Interactive chat
async def main():
    history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ('quit', 'exit'):
            break
        
        await chat_with_streaming(user_input, history)

asyncio.run(main())
```

## Performance Tips

### 1. Gunakan Streaming untuk Long Responses

```python
# Short responses - invoke cukup
short = chain.invoke({"topic": "AI in 1 sentence"})

# Long responses - stream
for chunk in chain.stream({"topic": "Write 1000 words about AI"}):
    yield chunk
```

### 2. Buffer untuk Batch Display

```python
import time

async def buffered_stream():
    buffer = []
    last_flush = time.time()
    
    async for chunk in chain.astream(input_data):
        buffer.append(chunk)
        
        # Flush every 100ms atau 10 chunks
        if time.time() - last_flush > 0.1 or len(buffer) >= 10:
            print("".join(buffer), end="", flush=True)
            buffer = []
            last_flush = time.time()
    
    # Flush remaining
    if buffer:
        print("".join(buffer), end="")
```

## Ringkasan

1. **`stream()`** - basic token streaming
2. **`astream()`** - async streaming untuk web apps
3. **`astream_events()`** - detailed events dari setiap component
4. **SSE** - standard untuk browser streaming
5. **Callbacks** - alternative untuk custom handling
6. **Collect while streaming** - dapatkan full response juga
7. **JSON streaming** - partial objects selama streaming

---

## 🎯 Use Case Modul 2: Smart Router

```python
#!/usr/bin/env python3
"""
Smart Router - Use Case Modul 2
Sistem yang menganalisis pertanyaan dan route ke chain yang tepat.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Specialized chains
math_chain = (
    ChatPromptTemplate.from_template("""
    Kamu adalah ahli matematika. Selesaikan dengan step-by-step.
    Problem: {input}
    """)
    | llm
    | StrOutputParser()
)

code_chain = (
    ChatPromptTemplate.from_template("""
    Kamu adalah senior programmer. Tulis clean code dengan comments.
    Request: {input}
    """)
    | llm
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("""
    Kamu adalah asisten yang helpful. Jawab dengan jelas dan informatif.
    Question: {input}
    """)
    | llm
    | StrOutputParser()
)

# Route classifier
class Route(BaseModel):
    category: Literal["math", "code", "general"] = Field(
        description="Category of the question"
    )

classifier = llm.with_structured_output(Route)

classify_chain = (
    ChatPromptTemplate.from_template(
        "Classify this question: {input}"
    )
    | classifier
)

# Router
def route_question(data: dict) -> str:
    category = data["route"].category
    input_text = data["input"]
    
    routes = {
        "math": math_chain,
        "code": code_chain,
        "general": general_chain
    }
    
    chain = routes.get(category, general_chain)
    return chain.invoke({"input": input_text})


# Full smart router with streaming
async def smart_router(question: str):
    print(f"\n📝 Question: {question}")
    
    # Classify
    route = await classify_chain.ainvoke({"input": question})
    print(f"🏷️  Category: {route.category}")
    print(f"\n🤖 Response: ", end="")
    
    # Route and stream
    routes = {"math": math_chain, "code": code_chain, "general": general_chain}
    selected_chain = routes.get(route.category, general_chain)
    
    async for chunk in selected_chain.astream({"input": question}):
        print(chunk, end="", flush=True)
    
    print("\n")


async def main():
    questions = [
        "What is 25 * 17?",
        "Write a Python function to reverse a string",
        "What is the capital of France?"
    ]
    
    for q in questions:
        await smart_router(q)
        print("-" * 50)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Selamat!** 🎉 Kamu sudah menyelesaikan Modul 2: LCEL!

---

**Selanjutnya:** [Modul 3: Prompt Engineering](/docs/prompt-engineering/prinsip-prompting) - Menulis prompt yang efektif.
