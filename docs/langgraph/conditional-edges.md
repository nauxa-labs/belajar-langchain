---
sidebar_position: 3
title: Conditional Edges
description: Routing dinamis berdasarkan state
---

# Conditional Edges

Conditional edges membuat graph **dynamic** - flow berubah berdasarkan state.

## Basic Conditional Edge

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class State(TypedDict):
    score: int
    result: str

def evaluate(state: State) -> dict:
    score = state["score"]
    if score >= 80:
        return {"result": "excellent"}
    elif score >= 60:
        return {"result": "pass"}
    else:
        return {"result": "fail"}

def route_by_result(state: State) -> Literal["celebrate", "retry", "give_up"]:
    result = state["result"]
    if result == "excellent":
        return "celebrate"
    elif result == "pass":
        return "retry"
    else:
        return "give_up"

# Build
builder = StateGraph(State)

builder.add_node("evaluate", evaluate)
builder.add_node("celebrate", lambda s: {"result": "🎉 Excellent!"})
builder.add_node("retry", lambda s: {"result": "📚 Study more"})
builder.add_node("give_up", lambda s: {"result": "😢 Try next time"})

builder.add_edge(START, "evaluate")
builder.add_conditional_edges(
    "evaluate",
    route_by_result,
    {
        "celebrate": "celebrate",
        "retry": "retry",
        "give_up": "give_up"
    }
)
builder.add_edge("celebrate", END)
builder.add_edge("retry", END)
builder.add_edge("give_up", END)

graph = builder.compile()
```

## Multiple Conditions

```python
def complex_router(state: State) -> str:
    """Multiple conditions for routing."""
    
    # Priority-based routing
    if state.get("error"):
        return "error_handler"
    
    if state.get("needs_human_review"):
        return "human_review"
    
    if state["confidence"] < 0.7:
        return "retry"
    
    if state["task_type"] == "research":
        return "research_agent"
    elif state["task_type"] == "writing":
        return "writing_agent"
    
    return "default_handler"

builder.add_conditional_edges(
    "classifier",
    complex_router,
    {
        "error_handler": "error_node",
        "human_review": "human_node",
        "retry": "retry_node",
        "research_agent": "research",
        "writing_agent": "writing",
        "default_handler": "default"
    }
)
```

## Loop with Exit Condition

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from operator import add

class LoopState(TypedDict):
    counter: int
    messages: Annotated[list[str], add]

def increment(state: LoopState) -> dict:
    new_count = state["counter"] + 1
    return {
        "counter": new_count,
        "messages": [f"Iteration {new_count}"]
    }

def should_continue(state: LoopState) -> Literal["continue", "__end__"]:
    if state["counter"] < 5:
        return "continue"
    return END

# Build loop
builder = StateGraph(LoopState)

builder.add_node("increment", increment)

builder.add_edge(START, "increment")
builder.add_conditional_edges(
    "increment",
    should_continue,
    {"continue": "increment"}  # Loop back
)

graph = builder.compile()

result = graph.invoke({"counter": 0, "messages": []})
print(f"Final count: {result['counter']}")  # 5
print(f"Messages: {result['messages']}")  # ['Iteration 1', ..., 'Iteration 5']
```

## Agent Decision Loop

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

llm_with_tools = llm.bind_tools([search])

def agent(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    
    # If there are tool calls, route to tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise end
    return END

def call_tools(state):
    last_message = state["messages"][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "search":
            result = search.invoke(tool_call["args"])
            results.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
    
    return {"messages": results}

# Build
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("tools", call_tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools"})
builder.add_edge("tools", "agent")  # Loop back to agent

graph = builder.compile()
```

## Parallel Branches

Send to multiple nodes simultaneously:

```python
from typing import Literal

def parallel_router(state: State) -> list[str]:
    """Return list of nodes to run in parallel."""
    return ["researcher", "writer", "critic"]

# Note: LangGraph handles parallel execution
builder.add_conditional_edges(
    "start",
    parallel_router,
    {
        "researcher": "research_node",
        "writer": "writing_node",
        "critic": "review_node"
    }
)
```

## Dynamic Routing with LLM

Let LLM decide the route:

```python
from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    next_step: Literal["search", "calculate", "respond"] = Field(
        description="Next action to take"
    )
    reasoning: str = Field(description="Why this choice")

def llm_router(state: State) -> str:
    router_prompt = """Based on the user's question, decide the next step:
    - "search": if need to find information online
    - "calculate": if need to do math
    - "respond": if can answer directly
    
    Question: {question}
    """
    
    chain = ChatPromptTemplate.from_template(router_prompt) | llm.with_structured_output(RouteDecision)
    
    decision = chain.invoke({"question": state["question"]})
    return decision.next_step

builder.add_conditional_edges(
    "classifier",
    llm_router,
    {
        "search": "search_node",
        "calculate": "calc_node",
        "respond": "response_node"
    }
)
```

## Error Handling Routes

```python
def safe_route(state: State) -> str:
    try:
        # Normal routing logic
        if state["task_complete"]:
            return "finish"
        return "process"
    except Exception as e:
        # Route to error handler
        return "error"

builder.add_conditional_edges(
    "main",
    safe_route,
    {
        "finish": END,
        "process": "process_node",
        "error": "error_handler"
    }
)
```

## Complete Example: Review Loop

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Literal
from operator import add

class ReviewState(TypedDict):
    content: str
    feedback: Annotated[list[str], add]
    revision_count: int
    approved: bool

llm = ChatOpenAI(model="gpt-4o-mini")

def write(state: ReviewState) -> dict:
    if state["feedback"]:
        prompt = f"Revise this content based on feedback:\n{state['content']}\n\nFeedback: {state['feedback'][-1]}"
    else:
        prompt = "Write a short paragraph about AI."
    
    response = llm.invoke(prompt)
    return {"content": response.content}

def review(state: ReviewState) -> dict:
    prompt = f"""Review this content and provide feedback.
If good, respond with just "APPROVED".
Otherwise, provide specific improvement suggestions.

Content: {state['content']}"""
    
    response = llm.invoke(prompt)
    feedback = response.content
    
    approved = "APPROVED" in feedback.upper()
    
    return {
        "feedback": [feedback],
        "revision_count": state["revision_count"] + 1,
        "approved": approved
    }

def should_continue(state: ReviewState) -> Literal["revise", "__end__"]:
    if state["approved"]:
        return END
    if state["revision_count"] >= 3:
        return END  # Max revisions
    return "revise"

# Build
builder = StateGraph(ReviewState)

builder.add_node("write", write)
builder.add_node("review", review)

builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", should_continue, {"revise": "write"})

graph = builder.compile()

# Run
result = graph.invoke({
    "content": "",
    "feedback": [],
    "revision_count": 0,
    "approved": False
})

print(f"Final content: {result['content']}")
print(f"Revisions: {result['revision_count']}")
print(f"Approved: {result['approved']}")
```

## Ringkasan

1. **add_conditional_edges()** - route based on state
2. **Routing function** returns next node name
3. **Loops** dengan exit condition
4. **LLM-based routing** untuk dynamic decisions
5. **Error handling** routes untuk robustness

---

**Selanjutnya:** [Checkpointing](/docs/langgraph/checkpointing) - Persistence dan state recovery.
