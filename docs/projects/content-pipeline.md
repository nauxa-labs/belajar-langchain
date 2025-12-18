---
sidebar_position: 5
title: "Proyek 4: Content Pipeline"
description: Multi-agent content creation dengan LangGraph
---

# Proyek 4: Content Pipeline dengan LangGraph

Membangun **multi-agent system** untuk content creation dengan **research**, **writing**, **review**, dan **human approval**.

## Requirements

### Fitur Utama
- ✅ Research phase (gather info)
- ✅ Writing phase (create draft)
- ✅ Review phase (quality check)
- ✅ Revision loop (improve based on feedback)
- ✅ Human approval gate
- ✅ Publishing via API

### Tech Stack
- LangGraph (state machine)
- Multi-agent collaboration
- Checkpointing untuk persistence
- Human-in-the-loop

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Content Creation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Topic Input                                                │
│        │                                                     │
│        ▼                                                     │
│   🔍 [RESEARCHER] ──▶ Research notes                        │
│        │                                                     │
│        ▼                                                     │
│   ✍️  [WRITER] ──▶ Draft content                             │
│        │                                                     │
│        ▼                                                     │
│   📝 [REVIEWER] ──▶ Feedback                                │
│        │                                                     │
│   ┌────┴────┐                                               │
│   ▼         ▼                                               │
│ REVISE   APPROVED                                           │
│   │         │                                               │
│   └──▶ WRITER   [HUMAN APPROVAL GATE]                       │
│                     │                                        │
│                ┌────┴────┐                                  │
│                ▼         ▼                                  │
│            PUBLISH    REJECT                                │
│                │                                             │
│                ▼                                             │
│            📤 [PUBLISHER] ──▶ API/CMS                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: State Definition

```python
from typing import TypedDict, Annotated, Literal
from operator import add

class ContentState(TypedDict):
    # Input
    topic: str
    content_type: str  # blog, social, newsletter
    
    # Research
    research_notes: str
    sources: Annotated[list[str], add]
    
    # Writing
    outline: str
    draft: str
    revision_count: int
    
    # Review
    feedback: Annotated[list[str], add]
    review_score: float
    
    # Status
    status: str
    approved_by_human: bool
    published: bool
    publish_url: str
```

### Step 2: Agent Nodes

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def researcher(state: ContentState) -> dict:
    """Research agent gathers information."""
    
    prompt = f"""Research the topic: {state['topic']}
Content type: {state['content_type']}

Provide:
1. Key facts and statistics
2. Current trends
3. Expert opinions
4. Relevant examples

Format as structured research notes."""

    response = llm.invoke(prompt)
    
    return {
        "research_notes": response.content,
        "sources": ["Research: " + state["topic"]],
        "status": "researched"
    }

def writer(state: ContentState) -> dict:
    """Writer creates or revises content."""
    
    if state.get("feedback"):
        # Revision mode
        prompt = f"""Revise this {state['content_type']}:

Current Draft:
{state['draft']}

Feedback to address:
{state['feedback'][-1]}

Research notes:
{state['research_notes']}

Create an improved version."""
    else:
        # Initial writing
        prompt = f"""Write a {state['content_type']} about: {state['topic']}

Research notes:
{state['research_notes']}

Requirements:
- Engaging introduction
- Well-structured body
- Clear conclusion
- Appropriate for the content type"""

    response = llm.invoke(prompt)
    
    return {
        "draft": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
        "status": "drafted"
    }

def reviewer(state: ContentState) -> dict:
    """Reviewer checks quality and provides feedback."""
    
    prompt = f"""Review this {state['content_type']}:

{state['draft']}

Evaluate:
1. Accuracy (based on research)
2. Clarity and structure
3. Engagement
4. Grammar and style

Give a score from 1-10 and specific feedback.
If score >= 8, say "APPROVED"
Otherwise, provide improvement suggestions."""

    response = llm.invoke(prompt)
    content = response.content
    
    # Parse score
    import re
    score_match = re.search(r'(\d+)/10', content)
    score = float(score_match.group(1)) if score_match else 5.0
    
    approved = "APPROVED" in content.upper() or score >= 8
    
    return {
        "feedback": [content],
        "review_score": score / 10,
        "status": "approved" if approved else "needs_revision"
    }

def publisher(state: ContentState) -> dict:
    """Publisher sends content to CMS/API."""
    
    # Simulate publishing
    import hashlib
    content_hash = hashlib.md5(state["draft"].encode()).hexdigest()[:8]
    url = f"https://example.com/content/{content_hash}"
    
    return {
        "published": True,
        "publish_url": url,
        "status": "published"
    }
```

### Step 3: Routing Logic

```python
from langgraph.graph import StateGraph, START, END

def after_review(state: ContentState) -> Literal["writer", "approval_gate"]:
    """Route based on review outcome."""
    
    if state["status"] == "needs_revision":
        if state["revision_count"] >= 3:
            # Max revisions reached, go to human
            return "approval_gate"
        return "writer"
    
    return "approval_gate"

def after_approval(state: ContentState) -> Literal["publisher", "__end__"]:
    """Route based on human approval."""
    
    if state.get("approved_by_human"):
        return "publisher"
    return END  # Rejected

def approval_gate(state: ContentState) -> dict:
    """Human approval checkpoint."""
    return {"status": "awaiting_approval"}
```

### Step 4: Build Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(ContentState)

# Add nodes
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_node("approval_gate", approval_gate)
builder.add_node("publisher", publisher)

# Add edges
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "reviewer")
builder.add_conditional_edges("reviewer", after_review)
builder.add_conditional_edges("approval_gate", after_approval)
builder.add_edge("publisher", END)

# Compile with checkpointing and human interrupt
pipeline = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approval_gate"]  # Pause for human
)
```

## Complete Code

```python
#!/usr/bin/env python3
"""
Content Creation Pipeline
Proyek 4 - Modul 10
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from operator import add
import re

load_dotenv()


# ===== STATE =====

class ContentState(TypedDict):
    topic: str
    content_type: str
    research_notes: str
    sources: Annotated[list[str], add]
    draft: str
    revision_count: int
    feedback: Annotated[list[str], add]
    review_score: float
    status: str
    approved_by_human: bool
    published: bool
    publish_url: str


# ===== AGENTS =====

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def researcher(state: ContentState) -> dict:
    prompt = f"Research '{state['topic']}' for a {state['content_type']}. Provide key facts, trends, examples."
    response = llm.invoke(prompt)
    return {
        "research_notes": response.content,
        "sources": [f"Research on: {state['topic']}"],
        "status": "researched"
    }

def writer(state: ContentState) -> dict:
    feedback = state.get("feedback", [])
    
    if feedback:
        prompt = f"""Revise based on feedback:

Draft: {state['draft'][:500]}...

Feedback: {feedback[-1][:200]}

Improve and rewrite."""
    else:
        prompt = f"""Write a {state['content_type']} about {state['topic']}.

Research: {state['research_notes'][:500]}"""

    response = llm.invoke(prompt)
    return {
        "draft": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
        "status": "drafted"
    }

def reviewer(state: ContentState) -> dict:
    prompt = f"""Review this {state['content_type']}:

{state['draft'][:800]}

Score 1-10 and feedback. Say APPROVED if >= 8."""

    response = llm.invoke(prompt)
    content = response.content
    
    score_match = re.search(r'(\d+)', content)
    score = float(score_match.group(1)) / 10 if score_match else 0.5
    approved = "APPROVED" in content.upper() or score >= 0.8
    
    return {
        "feedback": [content],
        "review_score": score,
        "status": "approved" if approved else "needs_revision"
    }

def approval_gate(state: ContentState) -> dict:
    return {"status": "awaiting_approval"}

def publisher(state: ContentState) -> dict:
    import hashlib
    url_hash = hashlib.md5(state["draft"].encode()).hexdigest()[:8]
    return {
        "published": True,
        "publish_url": f"https://blog.example.com/{url_hash}",
        "status": "published"
    }


# ===== ROUTING =====

def after_review(state: ContentState) -> Literal["writer", "approval_gate"]:
    if state["status"] == "needs_revision" and state["revision_count"] < 3:
        return "writer"
    return "approval_gate"

def after_approval(state: ContentState) -> Literal["publisher", "__end__"]:
    if state.get("approved_by_human"):
        return "publisher"
    return END


# ===== GRAPH =====

builder = StateGraph(ContentState)

builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_node("approval_gate", approval_gate)
builder.add_node("publisher", publisher)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "reviewer")
builder.add_conditional_edges("reviewer", after_review)
builder.add_conditional_edges("approval_gate", after_approval)
builder.add_edge("publisher", END)

pipeline = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approval_gate"]
)


# ===== MAIN =====

def create_content(topic: str, content_type: str = "blog post"):
    config = {"configurable": {"thread_id": f"content-{hash(topic)}"}}
    
    initial_state = {
        "topic": topic,
        "content_type": content_type,
        "research_notes": "",
        "sources": [],
        "draft": "",
        "revision_count": 0,
        "feedback": [],
        "review_score": 0.0,
        "status": "started",
        "approved_by_human": False,
        "published": False,
        "publish_url": ""
    }
    
    print(f"\n📝 Creating {content_type}: {topic}")
    print("=" * 50)
    
    # Run until approval gate
    result = pipeline.invoke(initial_state, config)
    
    print(f"\n📊 Status: {result['status']}")
    print(f"📝 Revisions: {result['revision_count']}")
    print(f"⭐ Score: {result['review_score']:.0%}")
    print(f"\n📄 Draft Preview:\n{result['draft'][:300]}...")
    
    # Human approval
    print("\n" + "=" * 50)
    approval = input("Approve for publishing? (yes/no): ").strip().lower()
    
    if approval == "yes":
        pipeline.update_state(config, {"approved_by_human": True})
        final = pipeline.invoke(None, config)
        print(f"\n✅ Published! URL: {final['publish_url']}")
        return final
    else:
        print("\n❌ Content rejected.")
        return result


def main():
    print("📰 Content Pipeline")
    print("Type a topic to create content, 'quit' to exit\n")
    
    while True:
        topic = input("Topic: ").strip()
        
        if topic.lower() in ('quit', 'exit'):
            break
        
        if not topic:
            continue
        
        content_type = input("Type (blog/social/newsletter) [blog]: ").strip() or "blog"
        create_content(topic, content_type)
        print()


if __name__ == "__main__":
    main()
```

## Usage

```bash
python content_pipeline.py

Topic: AI in Healthcare 2024
Type (blog/social/newsletter) [blog]: blog

📝 Creating blog: AI in Healthcare 2024
==================================================

📊 Status: awaiting_approval
📝 Revisions: 2
⭐ Score: 85%

📄 Draft Preview:
# AI in Healthcare: Transforming Patient Care in 2024...

==================================================
Approve for publishing? (yes/no): yes

✅ Published! URL: https://blog.example.com/a1b2c3d4
```

## Improvements

- [ ] Add image generation agent
- [ ] SEO optimization agent
- [ ] Social media adaptation
- [ ] Analytics integration
- [ ] A/B testing support

---

## 🎉 Selamat!

Kamu telah menyelesaikan **seluruh kurikulum Belajar LangChain**!

### Recap Perjalananmu:

| Modul | Apa yang Dipelajari |
|-------|---------------------|
| 0-4 | Fondasi: Models, Prompts, LCEL, Structured Output |
| 5 | RAG: Document Loading → Retrieval → Generation |
| 6 | Memory: Conversation history, persistence |
| 7 | Agents: Tools, function calling, patterns |
| 8 | LangGraph: State machines, multi-agent |
| 9 | Production: LangSmith, LangServe, best practices |
| 10 | Proyek: Real-world applications |

### Next Steps:

1. **Build your own projects** - mulai dari yang simple
2. **Explore LangChain docs** - masih banyak fitur lain
3. **Join community** - Discord, forums
4. **Contribute** - open source contributions

Selamat berkarya! 🚀
