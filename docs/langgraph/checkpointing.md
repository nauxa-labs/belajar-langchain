---
sidebar_position: 4
title: Checkpointing
description: Persistence, state recovery, dan time travel
---

# Checkpointing & Persistence

Checkpointing memungkinkan **save dan resume** state, plus **time travel debugging**.

## Mengapa Checkpointing?

```
Without Checkpoint:
User disconnects → All progress LOST

With Checkpoint:
User disconnects → Resume from last state ✅
```

Use cases:
- **Long-running workflows** (hours/days)
- **Human-in-the-loop** (async approval)
- **Debugging** (inspect any step)
- **Recovery** from failures

## MemorySaver (In-Memory)

Simplest checkpointer - good untuk development.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class State(TypedDict):
    messages: list[str]
    count: int

def step(state: State) -> dict:
    return {
        "messages": state["messages"] + [f"Step {state['count'] + 1}"],
        "count": state["count"] + 1
    }

# Build graph
builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

# Compile with checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Run with thread_id
config = {"configurable": {"thread_id": "user-123"}}

result1 = graph.invoke(
    {"messages": [], "count": 0},
    config=config
)
print(result1)  # {'messages': ['Step 1'], 'count': 1}

# Later - resume from checkpoint
result2 = graph.invoke(
    {"messages": result1["messages"], "count": result1["count"]},
    config=config
)
print(result2)  # {'messages': ['Step 1', 'Step 2'], 'count': 2}
```

## SQLite Checkpointer (Persistent)

Survives restart - good untuk production.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create SQLite checkpointer
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

graph = builder.compile(checkpointer=checkpointer)

# Now checkpoints persist to disk
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(initial_state, config=config)

# After restart, can resume!
```

## PostgreSQL Checkpointer

For production at scale:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)

graph = builder.compile(checkpointer=checkpointer)
```

```bash
pip install langgraph-checkpoint-postgres
```

## Accessing Checkpoint State

```python
# Get current state
state = graph.get_state(config)
print(state.values)  # Current state dict
print(state.next)  # Next node(s) to execute

# Get state history
for state in graph.get_state_history(config):
    print(f"Step: {state.metadata.get('step')}")
    print(f"Values: {state.values}")
```

## Time Travel Debugging

Go back to any previous state:

```python
# Get history
history = list(graph.get_state_history(config))

# Find a previous state
previous_state = history[2]  # 3rd checkpoint

# Resume from that state
result = graph.invoke(
    None,  # Use checkpoint state
    config={
        "configurable": {
            "thread_id": "user-123",
            "checkpoint_id": previous_state.config["configurable"]["checkpoint_id"]
        }
    }
)
```

## Modifying State

Update state manually:

```python
# Get current state
current = graph.get_state(config)

# Modify
graph.update_state(
    config,
    {"messages": current.values["messages"] + ["Manually added"]},
)

# Continue from modified state
result = graph.invoke(None, config=config)
```

## Complete Example: Resumable Workflow

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, Literal
from operator import add
import time

class WorkflowState(TypedDict):
    task: str
    steps_completed: Annotated[list[str], add]
    current_step: int
    total_steps: int

def process_step(state: WorkflowState) -> dict:
    step_num = state["current_step"]
    
    # Simulate work
    time.sleep(1)
    
    return {
        "steps_completed": [f"Completed step {step_num}"],
        "current_step": step_num + 1
    }

def should_continue(state: WorkflowState) -> Literal["process", "__end__"]:
    if state["current_step"] <= state["total_steps"]:
        return "process"
    return END

# Build
builder = StateGraph(WorkflowState)
builder.add_node("process", process_step)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", should_continue, {"process": "process"})

# Compile with SQLite persistence
checkpointer = SqliteSaver.from_conn_string("workflow.db")
graph = builder.compile(checkpointer=checkpointer)

def run_workflow(task: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check for existing state
    existing = graph.get_state(config)
    
    if existing.values:
        print(f"Resuming from step {existing.values['current_step']}")
        initial = None  # Use checkpoint
    else:
        print("Starting new workflow")
        initial = {
            "task": task,
            "steps_completed": [],
            "current_step": 1,
            "total_steps": 5
        }
    
    try:
        result = graph.invoke(initial, config=config)
        print(f"Completed! Steps: {result['steps_completed']}")
    except KeyboardInterrupt:
        print("Interrupted! Progress saved.")
        state = graph.get_state(config)
        print(f"Can resume from step {state.values['current_step']}")

# Usage
run_workflow("Process data", "workflow-001")

# If interrupted, run again to resume
# run_workflow("Process data", "workflow-001")
```

## Checkpoint Cleanup

```python
# Delete old checkpoints
from datetime import datetime, timedelta

async def cleanup_old_checkpoints(checkpointer, days=7):
    cutoff = datetime.now() - timedelta(days=days)
    
    # Implementation depends on checkpointer
    # For Postgres:
    await checkpointer.pool.execute(
        "DELETE FROM checkpoints WHERE created_at < $1",
        cutoff
    )
```

## Best Practices

### 1. Use Thread IDs Consistently

```python
# ✅ Good - unique per conversation
config = {"configurable": {"thread_id": f"user-{user_id}-conv-{conv_id}"}}

# ❌ Bad - reusing same thread
config = {"configurable": {"thread_id": "default"}}
```

### 2. Handle Missing Checkpoints

```python
def safe_invoke(graph, config, initial_state):
    existing = graph.get_state(config)
    
    if existing.values:
        return graph.invoke(None, config=config)
    else:
        return graph.invoke(initial_state, config=config)
```

### 3. Checkpoint Selectively

```python
# Not every node needs checkpoint
# Use for important milestones only

builder.add_node("quick_step", quick_step)  # No checkpoint needed
builder.add_node("important_step", important_step, checkpoint="before")
```

## Ringkasan

1. **MemorySaver** - development, in-memory
2. **SqliteSaver** - persistent, single server
3. **PostgresSaver** - production, scalable
4. **get_state()** - access current checkpoint
5. **get_state_history()** - time travel
6. **update_state()** - manual modification

---

**Selanjutnya:** [Human-in-the-Loop](/docs/langgraph/human-in-loop) - Interrupt untuk human approval.
