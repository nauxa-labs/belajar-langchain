---
sidebar_position: 7
title: Subgraphs
description: Composing reusable graph components
---

# Subgraphs

Subgraphs memungkinkan **modular, reusable** graph components.

## Kenapa Subgraphs?

```
Complex System:
┌─────────────────────────────────────────┐
│  Main Graph                              │
│  ┌─────────┐    ┌─────────────────────┐ │
│  │ Router  │───▶│  Research Subgraph  │ │
│  └─────────┘    └─────────────────────┘ │
│       │         ┌─────────────────────┐ │
│       └────────▶│  Writing Subgraph   │ │
│                 └─────────────────────┘ │
└─────────────────────────────────────────┘
```

Benefits:
- **Reusability** - use same subgraph in multiple places
- **Modularity** - develop/test independently
- **Clarity** - manage complexity

## Creating a Subgraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Subgraph State (can be different from parent)
class ResearchState(TypedDict):
    query: str
    sources: list[str]
    summary: str

def search(state: ResearchState) -> dict:
    # Simulate search
    return {"sources": [f"Source about {state['query']}"]}

def summarize(state: ResearchState) -> dict:
    sources = state["sources"]
    return {"summary": f"Summary of {len(sources)} sources"}

# Build subgraph
research_builder = StateGraph(ResearchState)
research_builder.add_node("search", search)
research_builder.add_node("summarize", summarize)
research_builder.add_edge(START, "search")
research_builder.add_edge("search", "summarize")
research_builder.add_edge("summarize", END)

research_subgraph = research_builder.compile()
```

## Using Subgraph as Node

```python
from typing import TypedDict

# Main graph state
class MainState(TypedDict):
    task: str
    research_query: str
    research_summary: str
    final_output: str

def prepare_research(state: MainState) -> dict:
    return {"research_query": f"Research for: {state['task']}"}

def run_research(state: MainState) -> dict:
    """Node that runs the research subgraph."""
    
    # Invoke subgraph with mapped state
    research_result = research_subgraph.invoke({
        "query": state["research_query"],
        "sources": [],
        "summary": ""
    })
    
    return {"research_summary": research_result["summary"]}

def generate_output(state: MainState) -> dict:
    return {"final_output": f"Output based on: {state['research_summary']}"}

# Main graph
main_builder = StateGraph(MainState)
main_builder.add_node("prepare", prepare_research)
main_builder.add_node("research", run_research)
main_builder.add_node("generate", generate_output)

main_builder.add_edge(START, "prepare")
main_builder.add_edge("prepare", "research")
main_builder.add_edge("research", "generate")
main_builder.add_edge("generate", END)

main_graph = main_builder.compile()

# Run
result = main_graph.invoke({"task": "Write about AI", "research_query": "", "research_summary": "", "final_output": ""})
print(result["final_output"])
```

## Nested Subgraphs

Subgraphs can contain subgraphs:

```python
# Level 1: Search subgraph
search_builder = StateGraph(SearchState)
# ... add nodes and edges
search_subgraph = search_builder.compile()

# Level 2: Research subgraph (uses search)
def search_node(state):
    result = search_subgraph.invoke({"query": state["query"]})
    return {"search_results": result["results"]}

research_builder = StateGraph(ResearchState)
research_builder.add_node("search", search_node)
research_builder.add_node("analyze", analyze_node)
# ... 
research_subgraph = research_builder.compile()

# Level 3: Main graph (uses research)
def research_node(state):
    result = research_subgraph.invoke({...})
    return {...}

main_builder = StateGraph(MainState)
main_builder.add_node("research", research_node)
# ...
```

## State Mapping

When subgraph has different state schema:

```python
class ParentState(TypedDict):
    user_query: str
    context: str
    response: str

class ChildState(TypedDict):
    input_text: str
    output_text: str

child_graph = build_child_graph()

def child_node(state: ParentState) -> dict:
    """Map parent state to child, run, map back."""
    
    # Map to child state
    child_input = {
        "input_text": state["user_query"],
        "output_text": ""
    }
    
    # Run child
    child_result = child_graph.invoke(child_input)
    
    # Map back to parent state
    return {"response": child_result["output_text"]}
```

## Conditional Subgraph Selection

```python
def route_to_subgraph(state: MainState) -> str:
    task_type = state.get("task_type", "general")
    
    if task_type == "research":
        return "research_subgraph"
    elif task_type == "writing":
        return "writing_subgraph"
    else:
        return "general_subgraph"

def run_research_subgraph(state):
    return research_graph.invoke(map_state(state))

def run_writing_subgraph(state):
    return writing_graph.invoke(map_state(state))

def run_general_subgraph(state):
    return general_graph.invoke(map_state(state))

builder.add_node("research_subgraph", run_research_subgraph)
builder.add_node("writing_subgraph", run_writing_subgraph)
builder.add_node("general_subgraph", run_general_subgraph)

builder.add_conditional_edges("router", route_to_subgraph)
```

## Reusable Component Library

```python
# subgraphs/research.py
from langgraph.graph import StateGraph, START, END

def create_research_subgraph(llm):
    """Factory function for research subgraph."""
    
    class ResearchState(TypedDict):
        topic: str
        findings: list[str]
    
    def search(state):
        response = llm.invoke(f"Research: {state['topic']}")
        return {"findings": [response.content]}
    
    builder = StateGraph(ResearchState)
    builder.add_node("search", search)
    builder.add_edge(START, "search")
    builder.add_edge("search", END)
    
    return builder.compile()

# subgraphs/writing.py
def create_writing_subgraph(llm):
    """Factory function for writing subgraph."""
    # Similar pattern...
    pass

# main.py
from subgraphs.research import create_research_subgraph
from subgraphs.writing import create_writing_subgraph

llm = ChatOpenAI(model="gpt-4o-mini")

research = create_research_subgraph(llm)
writing = create_writing_subgraph(llm)

# Use in main graph
```

## Complete Example: Pipeline with Subgraphs

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from operator import add

llm = ChatOpenAI(model="gpt-4o-mini")

# ===== RESEARCH SUBGRAPH =====
class ResearchState(TypedDict):
    topic: str
    facts: Annotated[list[str], add]

def gather_facts(state: ResearchState) -> dict:
    response = llm.invoke(f"List 3 key facts about: {state['topic']}")
    facts = response.content.split("\n")
    return {"facts": facts}

def verify_facts(state: ResearchState) -> dict:
    response = llm.invoke(f"Verify these facts:\n{state['facts']}")
    return {"facts": [f"[VERIFIED] {state['facts']}"]}

research_builder = StateGraph(ResearchState)
research_builder.add_node("gather", gather_facts)
research_builder.add_node("verify", verify_facts)
research_builder.add_edge(START, "gather")
research_builder.add_edge("gather", "verify")
research_builder.add_edge("verify", END)
research_subgraph = research_builder.compile()

# ===== WRITING SUBGRAPH =====
class WritingState(TypedDict):
    topic: str
    context: str
    draft: str

def outline(state: WritingState) -> dict:
    response = llm.invoke(f"Create outline for: {state['topic']}")
    return {"draft": response.content}

def write_draft(state: WritingState) -> dict:
    prompt = f"""Write content based on:
Topic: {state['topic']}
Context: {state['context']}
Outline: {state['draft']}"""
    response = llm.invoke(prompt)
    return {"draft": response.content}

writing_builder = StateGraph(WritingState)
writing_builder.add_node("outline", outline)
writing_builder.add_node("write", write_draft)
writing_builder.add_edge(START, "outline")
writing_builder.add_edge("outline", "write")
writing_builder.add_edge("write", END)
writing_subgraph = writing_builder.compile()

# ===== MAIN GRAPH =====
class MainState(TypedDict):
    task: str
    research_output: str
    writing_output: str
    final_result: str

def do_research(state: MainState) -> dict:
    result = research_subgraph.invoke({
        "topic": state["task"],
        "facts": []
    })
    return {"research_output": "\n".join(result["facts"])}

def do_writing(state: MainState) -> dict:
    result = writing_subgraph.invoke({
        "topic": state["task"],
        "context": state["research_output"],
        "draft": ""
    })
    return {"writing_output": result["draft"]}

def compile_final(state: MainState) -> dict:
    return {"final_result": f"""
=== RESEARCH ===
{state['research_output']}

=== CONTENT ===
{state['writing_output']}
"""}

main_builder = StateGraph(MainState)
main_builder.add_node("research", do_research)
main_builder.add_node("writing", do_writing)
main_builder.add_node("compile", compile_final)

main_builder.add_edge(START, "research")
main_builder.add_edge("research", "writing")
main_builder.add_edge("writing", "compile")
main_builder.add_edge("compile", END)

main_graph = main_builder.compile()

# Run
result = main_graph.invoke({
    "task": "Artificial Intelligence in Healthcare",
    "research_output": "",
    "writing_output": "",
    "final_result": ""
})

print(result["final_result"])
```

## Ringkasan

1. **Subgraphs** - compiled graphs used as nodes
2. **State mapping** - transform between parent/child states
3. **Factory functions** - create configurable subgraphs
4. **Nested subgraphs** - multiple levels of composition
5. **Reusable components** - modular graph library

---

**Selanjutnya:** [LangGraph Studio](/docs/langgraph/studio) - Visual debugging dan development.
