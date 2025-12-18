---
sidebar_position: 5
title: Human-in-the-Loop
description: Interrupt untuk human approval dan input
---

# Human-in-the-Loop

LangGraph memungkinkan **pause workflow** untuk menunggu human approval atau input.

## Interrupt Before/After Node

### Interrupt Before

Pause **sebelum** node dijalankan:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Build graph
builder = StateGraph(State)
builder.add_node("analyze", analyze)
builder.add_node("publish", publish)  # Need approval
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "publish")
builder.add_edge("publish", END)

# Compile with interrupt
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["publish"]  # Pause before publish
)
```

### Interrupt After

Pause **setelah** node dijalankan:

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_after=["analyze"]  # Pause after analyze
)
```

## Basic Human Approval Flow

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class ContentState(TypedDict):
    draft: str
    approved: bool
    feedback: str

def generate_draft(state: ContentState) -> dict:
    return {"draft": "This is the AI-generated draft content."}

def publish(state: ContentState) -> dict:
    return {"draft": state["draft"] + "\n[PUBLISHED]"}

# Build
builder = StateGraph(ContentState)
builder.add_node("generate", generate_draft)
builder.add_node("publish", publish)
builder.add_edge(START, "generate")
builder.add_edge("generate", "publish")
builder.add_edge("publish", END)

# Compile with interrupt before publish
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["publish"]
)

# Run until interrupt
config = {"configurable": {"thread_id": "content-001"}}

result = graph.invoke(
    {"draft": "", "approved": False, "feedback": ""},
    config=config
)

# Check state - paused before 'publish'
state = graph.get_state(config)
print(f"Draft: {state.values['draft']}")
print(f"Next: {state.next}")  # ('publish',)

# Human reviews and approves...
# Then resume
final = graph.invoke(None, config=config)
print(final["draft"])  # Contains [PUBLISHED]
```

## With Human Modification

Human dapat **modify state** sebelum resume:

```python
# After interrupt, human reviews
state = graph.get_state(config)
print(f"Draft to review:\n{state.values['draft']}")

# Human makes changes
human_feedback = input("Approve? (yes/no/edit): ")

if human_feedback == "yes":
    # Resume as-is
    result = graph.invoke(None, config=config)
    
elif human_feedback == "edit":
    # Modify state before resume
    new_draft = input("Enter revised draft: ")
    
    graph.update_state(
        config,
        {"draft": new_draft}
    )
    
    result = graph.invoke(None, config=config)
    
else:
    # Don't resume
    print("Content rejected")
```

## Approval Gateway Pattern

```python
from typing import TypedDict, Literal

class ApprovalState(TypedDict):
    content: str
    approval_status: Literal["pending", "approved", "rejected"]
    revision_notes: str

def create_content(state: ApprovalState) -> dict:
    return {
        "content": "Generated content here",
        "approval_status": "pending"
    }

def check_approval(state: ApprovalState) -> Literal["publish", "revise", "abort"]:
    status = state["approval_status"]
    
    if status == "approved":
        return "publish"
    elif status == "pending":
        return "revise"  # Not yet approved, revise
    else:
        return "abort"

def publish(state: ApprovalState) -> dict:
    return {"content": state["content"] + " [PUBLISHED]"}

def revise(state: ApprovalState) -> dict:
    notes = state.get("revision_notes", "")
    return {"content": f"Revised based on: {notes}"}

def abort(state: ApprovalState) -> dict:
    return {"content": "ABORTED"}

# Build
builder = StateGraph(ApprovalState)

builder.add_node("create", create_content)
builder.add_node("publish", publish)
builder.add_node("revise", revise)
builder.add_node("abort", abort)

builder.add_edge(START, "create")
builder.add_conditional_edges("create", check_approval, {
    "publish": "publish",
    "revise": "revise",
    "abort": "abort"
})
builder.add_edge("revise", "create")  # Loop back
builder.add_edge("publish", END)
builder.add_edge("abort", END)

# Compile with interrupt after create
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["create"]  # Wait for human approval
)

# Workflow
config = {"configurable": {"thread_id": "approval-001"}}

# First run - creates content, then pauses
graph.invoke({"content": "", "approval_status": "pending", "revision_notes": ""}, config)

# Human reviews
state = graph.get_state(config)
print(f"Review this: {state.values['content']}")

# Human approves
graph.update_state(config, {"approval_status": "approved"})

# Resume
result = graph.invoke(None, config)
print(result["content"])  # Published!
```

## Multi-Step Human Review

```python
from typing import TypedDict, List, Literal

class MultiReviewState(TypedDict):
    document: str
    reviews: List[dict]
    current_reviewer: str
    all_approved: bool

reviewers = ["legal", "marketing", "ceo"]

def assign_reviewer(state: MultiReviewState) -> dict:
    pending = [r for r in reviewers if r not in [rev["by"] for rev in state["reviews"]]]
    
    if pending:
        return {"current_reviewer": pending[0]}
    return {"current_reviewer": "", "all_approved": True}

def collect_review(state: MultiReviewState) -> dict:
    # This would be updated by human
    return {}

def route_after_review(state: MultiReviewState) -> Literal["assign", "__end__"]:
    if state.get("all_approved"):
        return END
    return "assign"

# Build
builder = StateGraph(MultiReviewState)

builder.add_node("assign", assign_reviewer)
builder.add_node("collect", collect_review)

builder.add_edge(START, "assign")
builder.add_edge("assign", "collect")
builder.add_conditional_edges("collect", route_after_review, {"assign": "assign"})

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["assign"]  # Pause for each reviewer
)

# Run
config = {"configurable": {"thread_id": "multi-review-001"}}

# Start
graph.invoke({
    "document": "Contract draft...",
    "reviews": [],
    "current_reviewer": "",
    "all_approved": False
}, config)

# For each reviewer
while True:
    state = graph.get_state(config)
    
    if state.next == ():  # No more nodes
        break
    
    reviewer = state.values["current_reviewer"]
    print(f"Waiting for {reviewer} review...")
    
    # Simulate human review
    review = {"by": reviewer, "status": "approved", "notes": "Looks good"}
    
    # Update state with review
    graph.update_state(config, {
        "reviews": state.values["reviews"] + [review]
    })
    
    # Continue
    graph.invoke(None, config)

print("All reviews complete!")
```

## Timeout Handling

```python
import asyncio
from datetime import datetime, timedelta

async def wait_for_approval(graph, config, timeout_hours=24):
    deadline = datetime.now() + timedelta(hours=timeout_hours)
    
    while datetime.now() < deadline:
        state = graph.get_state(config)
        
        if state.values.get("approved"):
            # Resume
            return await graph.ainvoke(None, config)
        
        # Check again in 5 minutes
        await asyncio.sleep(300)
    
    # Timeout - auto-reject
    graph.update_state(config, {"status": "timeout"})
    return graph.invoke(None, config)
```

## Web API Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
graphs = {}  # Store graph instances

class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    notes: str = ""

@app.post("/workflows/{thread_id}/start")
async def start_workflow(thread_id: str, content: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await graph.ainvoke(
        {"content": content, "status": "pending"},
        config
    )
    
    state = graph.get_state(config)
    
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval",
        "content": state.values["content"]
    }

@app.post("/workflows/{thread_id}/approve")
async def approve_workflow(thread_id: str, request: ApprovalRequest):
    config = {"configurable": {"thread_id": thread_id}}
    
    state = graph.get_state(config)
    if not state.values:
        raise HTTPException(404, "Workflow not found")
    
    # Update state
    graph.update_state(config, {
        "approved": request.approved,
        "notes": request.notes
    })
    
    # Resume
    result = await graph.ainvoke(None, config)
    
    return {"status": "completed", "result": result}

@app.get("/workflows/{thread_id}/status")
async def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    return {
        "values": state.values,
        "next": state.next
    }
```

## Ringkasan

1. **interrupt_before** - pause before node
2. **interrupt_after** - pause after node
3. **get_state()** - check paused state
4. **update_state()** - modify before resume
5. **invoke(None)** - resume from checkpoint
6. **API integration** untuk async approval

---

**Selanjutnya:** [Multi-Agent](/docs/langgraph/multi-agent) - Kolaborasi antar agents.
