---
sidebar_position: 7
title: Streaming Agents
description: Building responsive agent UIs dengan streaming
---

# Streaming Agents

Streaming membuat agent terasa **lebih responsif** dengan menampilkan output secara real-time.

## Why Streaming?

```
Without Streaming:
User waits... waits... waits... [FULL RESPONSE]

With Streaming:
User sees: "Let" → "me" → "search" → "for" → "that..." [REAL-TIME]
```

## Basic Streaming

### Stream Final Output Only

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Sync streaming
for chunk in executor.stream({"input": "What's the weather in Tokyo?"}):
    if "output" in chunk:
        print(chunk["output"], end="", flush=True)
print()
```

### Async Streaming

```python
async def stream_response(query: str):
    async for chunk in executor.astream({"input": query}):
        if "output" in chunk:
            print(chunk["output"], end="", flush=True)

import asyncio
asyncio.run(stream_response("Tell me a joke"))
```

## Streaming with Intermediate Steps

See what agent is doing in real-time:

```python
for chunk in executor.stream({"input": query}):
    # Agent action (tool call)
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"🔧 Using tool: {action.tool}")
            print(f"   Input: {action.tool_input}")
    
    # Tool result
    if "steps" in chunk:
        for step in chunk["steps"]:
            print(f"📊 Result: {step.observation[:100]}...")
    
    # Final output
    if "output" in chunk:
        print(f"✅ Answer: {chunk['output']}")
```

## astream_events (Detailed)

Most granular control over streaming:

```python
async def detailed_stream(query: str):
    async for event in executor.astream_events(
        {"input": query},
        version="v2"
    ):
        kind = event["event"]
        
        # Agent started
        if kind == "on_chain_start":
            if event["name"] == "AgentExecutor":
                print("🤖 Agent starting...")
        
        # LLM thinking
        if kind == "on_chat_model_start":
            print("🧠 Thinking...")
        
        # LLM streaming tokens
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        
        # Tool called
        if kind == "on_tool_start":
            print(f"\n🔧 Calling: {event['name']}")
            print(f"   Args: {event['data']['input']}")
        
        # Tool finished
        if kind == "on_tool_end":
            output = event["data"]["output"]
            print(f"   Result: {output[:200]}...")
        
        # Agent finished
        if kind == "on_chain_end":
            if event["name"] == "AgentExecutor":
                print("\n✅ Done!")

asyncio.run(detailed_stream("Search for AI news and summarize"))
```

## Event Types Reference

| Event | When |
|-------|------|
| `on_chain_start` | Chain/agent starts |
| `on_chain_end` | Chain/agent ends |
| `on_chat_model_start` | LLM starts generating |
| `on_chat_model_stream` | LLM streams token |
| `on_chat_model_end` | LLM finishes |
| `on_tool_start` | Tool called |
| `on_tool_end` | Tool finished |

## Building Streaming UI

### Terminal UI

```python
import sys

async def terminal_ui(query: str):
    print(f"{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}\n")
    
    current_tool = None
    
    async for event in executor.astream_events({"input": query}, version="v2"):
        kind = event["event"]
        
        if kind == "on_tool_start":
            current_tool = event["name"]
            print(f"\n⚙️  [{current_tool}] ", end="")
            sys.stdout.flush()
        
        elif kind == "on_tool_end":
            print("✓")
            current_tool = None
        
        elif kind == "on_chat_model_stream":
            if not current_tool:  # Only print final response tokens
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
    
    print(f"\n{'='*50}")
```

### FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/chat/stream")
async def stream_chat(query: Query):
    async def generate():
        async for event in executor.astream_events(
            {"input": query.message},
            version="v2"
        ):
            kind = event["event"]
            
            if kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name']})}\n\n"
            
            elif kind == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'result': event['data']['output'][:100]})}\n\n"
            
            elif kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### JavaScript Client

```javascript
async function streamChat(message) {
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\n\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                switch (data.type) {
                    case 'tool_start':
                        addToolIndicator(data.name);
                        break;
                    case 'token':
                        appendToResponse(data.content);
                        break;
                    case 'done':
                        finishResponse();
                        break;
                }
            }
        }
    }
}
```

## Complete Streaming Agent Example

```python
#!/usr/bin/env python3
"""
Streaming Research Agent with Rich UI
"""

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # Simulated delay
    import time
    time.sleep(1)
    return f"Found: {query} - Latest information from the web..."

@tool
def search_wikipedia(topic: str) -> str:
    """Search Wikipedia for factual information."""
    import time
    time.sleep(0.5)
    return f"Wikipedia: {topic} - Detailed factual information..."

@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [search_web, search_wikipedia, calculate]

# LLM with streaming
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant.
Use tools to gather information and provide comprehensive answers.
Always cite your sources."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


async def research(query: str):
    """Run research with streaming output."""
    print("\n" + "="*60)
    print(f"🔬 Research Query: {query}")
    print("="*60 + "\n")
    
    tool_count = 0
    
    async for event in executor.astream_events(
        {"input": query},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_tool_start":
            tool_count += 1
            tool_name = event["name"]
            tool_input = event["data"]["input"]
            print(f"🔧 [{tool_count}] Using {tool_name}...")
            print(f"    Query: {str(tool_input)[:50]}...")
        
        elif kind == "on_tool_end":
            output = event["data"]["output"]
            print(f"    ✓ Got result ({len(output)} chars)\n")
        
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
    
    print("\n\n" + "="*60)
    print(f"✅ Research complete! Used {tool_count} tools.")
    print("="*60 + "\n")


async def interactive_session():
    """Interactive streaming session."""
    print("\n🤖 Research Assistant (type 'quit' to exit)\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        await research(query)


if __name__ == "__main__":
    # Single query
    # asyncio.run(research("What is the latest news about AI?"))
    
    # Interactive mode
    asyncio.run(interactive_session())
```

## Performance Tips

### 1. Use Async

```python
# ✅ Better - non-blocking
async for chunk in executor.astream({"input": query}):
    process(chunk)

# ❌ Blocking
for chunk in executor.stream({"input": query}):
    process(chunk)
```

### 2. Buffer Tokens

```python
buffer = []
buffer_size = 5

async for event in executor.astream_events({"input": query}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            buffer.append(content)
            
            if len(buffer) >= buffer_size:
                yield "".join(buffer)
                buffer = []

# Flush remaining
if buffer:
    yield "".join(buffer)
```

### 3. Handle Disconnects

```python
async def safe_stream(query: str):
    try:
        async for event in executor.astream_events({"input": query}, version="v2"):
            yield event
    except asyncio.CancelledError:
        print("Client disconnected")
        raise
    except Exception as e:
        yield {"event": "error", "data": str(e)}
```

## Use Case Modul 7: Research Assistant

Fitur:
- ✅ Multiple search tools
- ✅ Real-time streaming
- ✅ Tool usage indicators
- ✅ Error handling
- ✅ Interactive mode

## Ringkasan

1. **stream()** - basic streaming
2. **astream_events()** - detailed event streaming
3. Stream **tool calls** dan **results** untuk transparency
4. Build **SSE endpoints** untuk web UIs
5. Use **async** untuk performance
6. Handle **disconnects** gracefully

---

**Selamat!** 🎉 Kamu sudah menyelesaikan **Modul 7: Agents & Tool Calling**!

Kamu sekarang bisa membuat:
- Agents dengan custom tools
- ReAct dan agent patterns lainnya
- Streaming responsive UIs

---

**Selanjutnya:** [Modul 8: LangGraph](/docs/langgraph/intro) - Multi-agent workflows dan state management.
