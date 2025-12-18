---
sidebar_position: 6
title: Multi-Agent
description: Kolaborasi antar agents dengan LangGraph
---

# Multi-Agent dengan LangGraph

Membangun sistem dimana **multiple agents collaborate** untuk menyelesaikan tasks kompleks.

## Multi-Agent Patterns

```
1. Supervisor Pattern
   Supervisor → delegates → Worker A/B/C

2. Peer Network
   Agent A ←→ Agent B ←→ Agent C

3. Hierarchical
   Manager → Supervisors → Workers
```

## Supervisor Pattern

Satu agent sebagai **supervisor** yang mendelegasi ke workers.

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Literal
from operator import add

# State
class TeamState(TypedDict):
    task: str
    messages: Annotated[list, add]
    next_worker: str
    result: str

# Agents
llm = ChatOpenAI(model="gpt-4o-mini")

def supervisor(state: TeamState) -> dict:
    """Supervisor decides who works next."""
    
    prompt = f"""You are a team supervisor. Given this task, decide which worker should handle it:
- researcher: for finding information
- writer: for creating content
- reviewer: for checking quality
- FINISH: if task is complete

Task: {state['task']}
Current progress: {state['messages']}

Reply with just the worker name or FINISH."""

    response = llm.invoke(prompt)
    next_worker = response.content.strip().lower()
    
    return {"next_worker": next_worker}

def researcher(state: TeamState) -> dict:
    """Research agent."""
    prompt = f"Research this topic: {state['task']}"
    response = llm.invoke(prompt)
    return {"messages": [f"Researcher: {response.content}"]}

def writer(state: TeamState) -> dict:
    """Writing agent."""
    context = "\n".join(state["messages"])
    prompt = f"Write content about {state['task']} using this research:\n{context}"
    response = llm.invoke(prompt)
    return {"messages": [f"Writer: {response.content}"]}

def reviewer(state: TeamState) -> dict:
    """Review agent."""
    context = "\n".join(state["messages"])
    prompt = f"Review this content and suggest improvements:\n{context}"
    response = llm.invoke(prompt)
    return {"messages": [f"Reviewer: {response.content}"]}

def route_to_worker(state: TeamState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    next_worker = state.get("next_worker", "")
    if next_worker == "finish" or not next_worker:
        return END
    return next_worker

# Build
builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_worker, {
    "researcher": "researcher",
    "writer": "writer",
    "reviewer": "reviewer"
})
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")

graph = builder.compile()

# Run
result = graph.invoke({
    "task": "Write a blog post about AI trends in 2024",
    "messages": [],
    "next_worker": "",
    "result": ""
})
```

## Structured Supervisor

Using structured output for better control:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SupervisorDecision(BaseModel):
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    reasoning: str = Field(description="Why this choice")

def supervisor(state: TeamState) -> dict:
    system = """You supervise a team of workers:
- researcher: finds information
- writer: creates content
- reviewer: checks quality
- FINISH: when task is complete

Analyze the current progress and decide next step."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"Task: {state['task']}\n\nProgress:\n{state['messages']}")
    ])
    
    chain = prompt | llm.with_structured_output(SupervisorDecision)
    decision = chain.invoke({})
    
    return {
        "next_worker": decision.next.lower(),
        "messages": [f"Supervisor: {decision.reasoning}"]
    }
```

## Tool-Based Multi-Agent

Each agent has different tools:

```python
from langchain_core.tools import tool

# Researcher tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def search_papers(query: str) -> str:
    """Search academic papers."""
    return f"Found papers about: {query}"

# Writer tools
@tool
def write_outline(topic: str) -> str:
    """Create content outline."""
    return f"Outline for: {topic}"

@tool
def write_section(section: str, context: str) -> str:
    """Write a section of content."""
    return f"Written section: {section}"

# Create specialized agents
researcher_agent = create_tool_calling_agent(
    llm, 
    [search_web, search_papers], 
    researcher_prompt
)

writer_agent = create_tool_calling_agent(
    llm,
    [write_outline, write_section],
    writer_prompt
)
```

## Agent Handoff

Explicit handoff between agents:

```python
from typing import TypedDict, Literal

class HandoffState(TypedDict):
    messages: list
    current_agent: str
    handoff_to: str
    handoff_reason: str

def agent_with_handoff(agent_name: str, next_agents: list[str]):
    """Create agent that can handoff to others."""
    
    def agent_fn(state: HandoffState) -> dict:
        system = f"""You are the {agent_name} agent.
When you complete your part, you can handoff to: {next_agents}
To handoff, say: HANDOFF TO [agent_name]: [reason]"""
        
        response = llm.invoke([
            {"role": "system", "content": system},
            *state["messages"]
        ])
        
        content = response.content
        
        # Check for handoff
        if "HANDOFF TO" in content:
            parts = content.split("HANDOFF TO")[1].split(":")
            next_agent = parts[0].strip().lower()
            reason = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "messages": [response],
                "handoff_to": next_agent,
                "handoff_reason": reason
            }
        
        return {"messages": [response], "handoff_to": ""}
    
    return agent_fn

# Create agents
researcher_fn = agent_with_handoff("researcher", ["writer", "reviewer"])
writer_fn = agent_with_handoff("writer", ["reviewer", "researcher"])
reviewer_fn = agent_with_handoff("reviewer", ["writer", "researcher"])
```

## Parallel Multi-Agent

Run multiple agents simultaneously:

```python
from langgraph.graph import StateGraph, START, END

class ParallelState(TypedDict):
    task: str
    research_result: str
    outline_result: str
    ready_to_combine: bool

def run_researcher(state: ParallelState) -> dict:
    result = llm.invoke(f"Research: {state['task']}")
    return {"research_result": result.content}

def run_outliner(state: ParallelState) -> dict:
    result = llm.invoke(f"Create outline for: {state['task']}")
    return {"outline_result": result.content}

def combine_results(state: ParallelState) -> dict:
    combined = f"""
Research: {state['research_result']}

Outline: {state['outline_result']}
"""
    final = llm.invoke(f"Combine this into a final document:\n{combined}")
    return {"result": final.content}

# Build with parallel branches
builder = StateGraph(ParallelState)

builder.add_node("researcher", run_researcher)
builder.add_node("outliner", run_outliner)
builder.add_node("combiner", combine_results)

# Parallel: both start from START
builder.add_edge(START, "researcher")
builder.add_edge(START, "outliner")

# Both lead to combiner
builder.add_edge("researcher", "combiner")
builder.add_edge("outliner", "combiner")
builder.add_edge("combiner", END)

graph = builder.compile()
```

## Complete Example: Content Team

```python
#!/usr/bin/env python3
"""
Multi-Agent Content Creation Team
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Literal
from operator import add
from pydantic import BaseModel, Field

# State
class ContentTeamState(TypedDict):
    topic: str
    messages: Annotated[list[str], add]
    research: str
    draft: str
    feedback: str
    iteration: int
    status: str

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Supervisor decision
class Assignment(BaseModel):
    agent: Literal["researcher", "writer", "reviewer", "DONE"]
    instruction: str

def supervisor(state: ContentTeamState) -> dict:
    """Supervisor coordinates the team."""
    
    prompt = f"""You manage a content creation team.
    
Topic: {state['topic']}
Current status: {state['status']}
Iteration: {state['iteration']}

Team progress:
{chr(10).join(state['messages'][-5:])}  # Last 5 messages

Decide next action:
- researcher: need more information
- writer: need to write/revise content
- reviewer: need quality check
- DONE: content is ready

Only say DONE if you have a reviewed, quality piece ready."""

    chain = ChatPromptTemplate.from_template(prompt) | llm.with_structured_output(Assignment)
    decision = chain.invoke({})
    
    return {
        "messages": [f"📋 Supervisor: Assigning to {decision.agent} - {decision.instruction}"],
        "status": decision.agent
    }

def researcher(state: ContentTeamState) -> dict:
    """Research agent gathers information."""
    
    prompt = f"Research the topic '{state['topic']}' and provide key facts, trends, and insights."
    response = llm.invoke(prompt)
    
    return {
        "research": response.content,
        "messages": [f"🔍 Researcher: Completed research on {state['topic']}"],
        "status": "researched"
    }

def writer(state: ContentTeamState) -> dict:
    """Writer creates content."""
    
    if state.get("feedback"):
        prompt = f"""Revise this draft based on feedback:

Draft: {state['draft']}

Feedback: {state['feedback']}

Research: {state['research']}"""
    else:
        prompt = f"""Write a compelling article about '{state['topic']}'.

Use this research:
{state['research']}"""

    response = llm.invoke(prompt)
    
    return {
        "draft": response.content,
        "messages": [f"✍️ Writer: {'Revised' if state.get('feedback') else 'Created'} draft"],
        "status": "drafted",
        "feedback": ""  # Clear old feedback
    }

def reviewer(state: ContentTeamState) -> dict:
    """Reviewer checks quality."""
    
    prompt = f"""Review this article for quality, accuracy, and engagement.

Article:
{state['draft']}

If good, say "APPROVED: [brief praise]"
If needs work, provide specific feedback."""

    response = llm.invoke(prompt)
    content = response.content
    
    approved = "APPROVED" in content.upper()
    
    return {
        "feedback": "" if approved else content,
        "messages": [f"📝 Reviewer: {'Approved!' if approved else 'Needs revision'}"],
        "status": "approved" if approved else "needs_revision",
        "iteration": state["iteration"] + 1
    }

def route(state: ContentTeamState) -> str:
    status = state.get("status", "")
    
    if status == "DONE" or state["iteration"] > 3:
        return END
    elif status in ["researcher"]:
        return "researcher"
    elif status in ["writer"]:
        return "writer"
    elif status in ["reviewer"]:
        return "reviewer"
    
    return "supervisor"

# Build
builder = StateGraph(ContentTeamState)

builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route)
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")

graph = builder.compile(checkpointer=MemorySaver())

# Run
def create_content(topic: str):
    config = {"configurable": {"thread_id": f"content-{hash(topic)}"}}
    
    result = graph.invoke({
        "topic": topic,
        "messages": [],
        "research": "",
        "draft": "",
        "feedback": "",
        "iteration": 0,
        "status": "started"
    }, config)
    
    print("\n=== FINAL RESULT ===")
    print(f"Topic: {result['topic']}")
    print(f"Iterations: {result['iteration']}")
    print(f"\nDraft:\n{result['draft'][:500]}...")
    
    return result

if __name__ == "__main__":
    create_content("The Future of Remote Work in 2025")
```

## Ringkasan

1. **Supervisor pattern** - one coordinator, multiple workers
2. **Tool-based agents** - specialized capabilities
3. **Handoff** - explicit agent transitions
4. **Parallel agents** - concurrent execution
5. **Iteration limits** - prevent infinite loops

---

**Selanjutnya:** [Subgraphs](/docs/langgraph/subgraphs) - Composing reusable graph components.
