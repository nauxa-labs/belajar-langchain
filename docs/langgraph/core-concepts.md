---
sidebar_position: 2
title: Konsep Inti
description: StateGraph, State, Nodes, dan Edges
---

# Konsep Inti LangGraph

Understanding the core building blocks of LangGraph.

## State

State adalah **container untuk semua data** yang mengalir melalui graph.

### Basic State (TypedDict)

```python
from typing import TypedDict

class State(TypedDict):
    messages: list[str]
    current_step: str
    iteration_count: int
```

### State with Reducers

Reducers menentukan **bagaimana updates digabung** ke state.

```python
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    # Messages akan di-append (not replaced)
    messages: Annotated[list[str], add]
    
    # This will be replaced
    current_step: str
```

### Pydantic State

For validation:

```python
from pydantic import BaseModel, Field
from typing import List

class AgentState(BaseModel):
    messages: List[str] = Field(default_factory=list)
    current_step: str = ""
    iteration: int = 0
    
    class Config:
        arbitrary_types_allowed = True
```

## Nodes

Nodes adalah **functions yang memodifikasi state**.

### Basic Node

```python
def my_node(state: State) -> dict:
    """A node that processes state and returns updates."""
    # Do something
    new_message = f"Processed at step {state['current_step']}"
    
    # Return updates (merged into state)
    return {"messages": [new_message]}
```

### Node with LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

def agent_node(state: State) -> dict:
    """Node that calls LLM."""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}
```

### Async Node

```python
async def async_node(state: State) -> dict:
    """Async node for I/O operations."""
    result = await fetch_data()
    return {"data": result}
```

## Edges

Edges menentukan **flow antar nodes**.

### Normal Edge

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# Add nodes
builder.add_node("node_a", node_a_func)
builder.add_node("node_b", node_b_func)

# Add edges
builder.add_edge(START, "node_a")  # Start → A
builder.add_edge("node_a", "node_b")  # A → B
builder.add_edge("node_b", END)  # B → End
```

### Conditional Edge

Route based on state:

```python
def routing_function(state: State) -> str:
    """Decide next node based on state."""
    if state["needs_review"]:
        return "review"
    else:
        return "publish"

# Add conditional edge
builder.add_conditional_edges(
    "write",  # From node
    routing_function,  # Function that returns next node name
    {
        "review": "review_node",
        "publish": "publish_node"
    }
)
```

## Building a Graph

Complete example:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

# 1. Define State
class ChatState(TypedDict):
    messages: Annotated[list[str], add]
    step: str

# 2. Define Nodes
def greet(state: ChatState) -> dict:
    return {
        "messages": ["Hello! How can I help?"],
        "step": "greeted"
    }

def process(state: ChatState) -> dict:
    # Get last user message
    user_msg = state["messages"][-1] if state["messages"] else ""
    response = f"You said: {user_msg}"
    return {
        "messages": [response],
        "step": "processed"
    }

def should_continue(state: ChatState) -> str:
    # Continue if less than 3 exchanges
    if len(state["messages"]) < 6:
        return "continue"
    return "end"

# 3. Build Graph
builder = StateGraph(ChatState)

builder.add_node("greet", greet)
builder.add_node("process", process)

builder.add_edge(START, "greet")
builder.add_edge("greet", "process")
builder.add_conditional_edges(
    "process",
    should_continue,
    {"continue": "process", "end": END}
)

# 4. Compile
graph = builder.compile()

# 5. Run
result = graph.invoke({
    "messages": ["Hi there!"],
    "step": ""
})

print(result)
```

## Graph with LLM Agent

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Literal
from operator import add

# State
class AgentState(TypedDict):
    messages: Annotated[list, add]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Node
def agent(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Router
def should_continue(state: AgentState) -> Literal["agent", "__end__"]:
    last_message = state["messages"][-1]
    
    # If AI wants to call a tool, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "agent"
    
    # Otherwise, end
    return END

# Build
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)

graph = builder.compile()

# Run
result = graph.invoke({
    "messages": [HumanMessage(content="Hello!")]
})
```

## State Updates

### Replacement (Default)

```python
class State(TypedDict):
    value: str  # Gets replaced

def node(state):
    return {"value": "new value"}  # Replaces old value
```

### Append (With Reducer)

```python
from operator import add

class State(TypedDict):
    items: Annotated[list, add]  # Gets appended

def node(state):
    return {"items": ["new item"]}  # Appends to list
```

### Custom Reducer

```python
def merge_dicts(existing: dict, new: dict) -> dict:
    return {**existing, **new}

class State(TypedDict):
    data: Annotated[dict, merge_dicts]
```

## Visualization

```python
# Generate Mermaid diagram
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# Or get Mermaid code
print(graph.get_graph().draw_mermaid())
```

## Best Practices

### 1. Keep State Minimal

```python
# ❌ Too much in state
class State(TypedDict):
    raw_data: bytes  # Large
    cached_results: dict  # Too much
    
# ✅ Essential only
class State(TypedDict):
    messages: list
    current_step: str
```

### 2. Pure Node Functions

```python
# ❌ Side effects
def bad_node(state):
    global counter  # Modifying global
    counter += 1
    return {"count": counter}

# ✅ Pure function
def good_node(state):
    return {"count": state["count"] + 1}
```

### 3. Clear Routing Logic

```python
# ❌ Complex inline logic
builder.add_conditional_edges(
    "node",
    lambda s: "a" if s["x"] > 5 and s["y"] < 10 else "b"
)

# ✅ Named function with clear logic
def route_decision(state) -> str:
    """Route based on business rules."""
    if state["x"] > 5 and state["y"] < 10:
        return "path_a"
    return "path_b"

builder.add_conditional_edges("node", route_decision)
```

## Ringkasan

1. **State** - typed container for data
2. **Nodes** - functions that transform state
3. **Edges** - connections (normal or conditional)
4. **Reducers** - how updates merge (replace vs append)
5. **Compile** - create runnable graph

---

**Selanjutnya:** [Conditional Edges](/docs/langgraph/conditional-edges) - Routing dinamis berdasarkan state.
