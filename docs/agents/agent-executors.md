---
sidebar_position: 5
title: Agent Executors
description: Menjalankan agents dengan execution loop
---

# Agent Executors

AgentExecutor adalah **engine yang menjalankan agent loop** - thinking, acting, observing, dan repeating.

## Basic Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentExecutor                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input ──▶ [Agent] ──▶ Decision                            │
│                │                                             │
│                ├──▶ Tool Call? ──▶ [Execute Tool]           │
│                │         │                                   │
│                │         ▼                                   │
│                │    Observation ──▶ [Agent] ──▶ ...         │
│                │                                             │
│                └──▶ Final Answer? ──▶ Output                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Creating Tool Calling Agent

The modern, recommended approach:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Define tools
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 25°C, sunny"

@tool
def calculate(expression: str) -> str:
    """Calculate math expression."""
    return str(eval(expression))

tools = [get_weather, calculate]

# 2. Create LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # Required for agent
])

# 4. Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. Create executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Run
result = executor.invoke({"input": "What's the weather in Tokyo?"})
print(result["output"])
```

## AgentExecutor Configuration

### Key Parameters

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    
    # Debugging
    verbose=True,  # Print step-by-step reasoning
    
    # Safety
    max_iterations=15,  # Max loops (default: 15)
    max_execution_time=60,  # Timeout in seconds
    
    # Error handling
    handle_parsing_errors=True,  # Recover from LLM errors
    
    # Output
    return_intermediate_steps=True,  # Include reasoning in output
    
    # Early stopping
    early_stopping_method="force",  # or "generate"
)
```

### Understanding Outputs

```python
result = executor.invoke({"input": "What's 10 + 20?"})

# Basic output
print(result["output"])  # Final answer

# With intermediate steps
print(result["intermediate_steps"])
# [
#   (AgentAction(tool='calculate', tool_input={'expression': '10 + 20'}), '30'),
# ]
```

## Handling Errors

### Parsing Errors

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  # Auto-recover
)

# Or custom handler
def custom_error_handler(error):
    return f"There was an error: {str(error)}. Please try again."

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=custom_error_handler
)
```

### Tool Errors

```python
from langchain_core.tools import ToolException

@tool
def risky_operation(data: str) -> str:
    """A tool that might fail."""
    if not data:
        raise ToolException("Data is required")
    return f"Processed: {data}"

# Executor will catch ToolException and let LLM retry
```

### Max Iterations

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10  # Stop after 10 loops
)

# If max reached, returns last state or error message
```

## With Memory

Add conversation memory to agent:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Prompt with history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember our conversation."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Add memory
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    executor,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Use with session
config = {"configurable": {"session_id": "user_123"}}
result = agent_with_memory.invoke(
    {"input": "Hi, my name is Alex"},
    config=config
)
```

## Streaming Agent Output

### Stream Final Output

```python
for chunk in executor.stream({"input": "What's the weather?"}):
    if "output" in chunk:
        print(chunk["output"], end="", flush=True)
```

### Stream All Events

```python
async for event in executor.astream_events(
    {"input": "Calculate 25 * 4"},
    version="v2"
):
    kind = event["event"]
    
    if kind == "on_chat_model_start":
        print("🧠 Agent thinking...")
    
    elif kind == "on_tool_start":
        print(f"🔧 Using tool: {event['name']}")
    
    elif kind == "on_tool_end":
        print(f"📊 Tool result: {event['data']['output'][:100]}")
    
    elif kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            print(content, end="", flush=True)
```

## Agent Types

### 1. Tool Calling Agent (Recommended)

```python
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)
```

Works with: OpenAI, Anthropic, Google, Mistral

### 2. ReAct Agent

```python
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

# ReAct needs specific prompt format
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)
```

### 3. Structured Chat Agent

For models without native function calling:

```python
from langchain.agents import create_structured_chat_agent

structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """Respond to the human as helpfully as possible.
You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
"""),
    ("human", "{input}\n\n{agent_scratchpad}")
])

agent = create_structured_chat_agent(llm, tools, structured_prompt)
```

## Complete Production Example

```python
#!/usr/bin/env python3
"""
Production-ready Agent with error handling and streaming.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated
    return f"Search results for '{query}': Found relevant information..."

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = [search_web, get_current_time, calculate]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools.

Guidelines:
1. Use tools when you need external information or calculations
2. Think step by step for complex problems
3. If a tool fails, try an alternative approach
4. Always provide helpful, accurate responses"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor with production settings
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Set True for debugging
    max_iterations=10,
    max_execution_time=30,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)


def run_agent(query: str) -> dict:
    """Run agent with error handling."""
    try:
        logging.info(f"Processing: {query}")
        result = executor.invoke({"input": query})
        
        return {
            "success": True,
            "output": result["output"],
            "steps": len(result.get("intermediate_steps", []))
        }
    except Exception as e:
        logging.error(f"Agent error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def stream_agent(query: str):
    """Stream agent response."""
    print(f"\n🤖 Processing: {query}\n")
    
    async for event in executor.astream_events(
        {"input": query},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_tool_start":
            print(f"🔧 Using: {event['name']}")
        elif kind == "on_tool_end":
            output = event['data']['output']
            print(f"   Result: {output[:100]}...")
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
    
    print("\n")


if __name__ == "__main__":
    # Sync example
    result = run_agent("What time is it and what's 42 * 3?")
    print(f"Result: {result}")
    
    # Async streaming
    import asyncio
    asyncio.run(stream_agent("Search for latest AI news"))
```

## Ringkasan

1. **AgentExecutor** runs the agent loop
2. **create_tool_calling_agent** - modern, recommended
3. **Configuration**: max_iterations, timeouts, error handling
4. **Memory** via RunnableWithMessageHistory
5. **Streaming** with astream_events
6. Always **handle errors** in production

---

**Selanjutnya:** [Agent Patterns](/docs/agents/agent-patterns) - ReAct, Plan-and-Execute, dan patterns lainnya.
