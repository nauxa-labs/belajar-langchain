---
sidebar_position: 1
title: LangSmith Setup
description: Tracing, monitoring, dan evaluation platform
---

# LangSmith Setup

LangSmith adalah **observability platform** untuk LangChain applications. Monitor, debug, dan evaluate LLM apps.

## Apa itu LangSmith?

```
┌─────────────────────────────────────────────────────────────┐
│                       LangSmith                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│   │ Tracing │  │  Debug  │  │ Evaluate│  │ Monitor │       │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│                                                              │
│   Features:                                                   │
│   ✅ Trace every LLM call                                    │
│   ✅ Debug issues visually                                   │
│   ✅ Run evaluations at scale                                │
│   ✅ Track latency & costs                                   │
│   ✅ Team collaboration                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

### 1. Create Account

Go to [smith.langchain.com](https://smith.langchain.com) and sign up.

### 2. Get API Key

1. Go to Settings → API Keys
2. Create new key
3. Copy and save securely

### 3. Environment Setup

```bash
# .env
LANGCHAIN_API_KEY=lsv2_pt_xxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="my-first-project"
```

### 4. Install SDK

```bash
pip install langsmith
```

## Basic Tracing

With environment variables set, tracing is **automatic**:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# This call is automatically traced!
response = llm.invoke("Hello, world!")
print(response.content)

# Check LangSmith dashboard to see the trace
```

## Project Organization

### Create Project

```python
from langsmith import Client

client = Client()

# Create new project
client.create_project("chatbot-production")
client.create_project("chatbot-staging")
client.create_project("experiments")
```

### Set Project per Environment

```python
import os

# Development
os.environ["LANGCHAIN_PROJECT"] = "chatbot-dev"

# Or in code
from langchain_core.tracers import LangChainTracer

tracer = LangChainTracer(project_name="chatbot-production")
llm.invoke("Hello", config={"callbacks": [tracer]})
```

## Run with Metadata

Add metadata for better organization:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    run_name="user-query-processing",
    tags=["production", "chatbot"],
    metadata={
        "user_id": "user-123",
        "session_id": "sess-456",
        "version": "1.0.0"
    }
)

response = llm.invoke("What is AI?", config=config)
```

## Tracing Complex Chains

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# Build chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Trace the entire chain
response = chain.invoke(
    {"input": "Explain quantum computing"},
    config={
        "run_name": "explain-topic",
        "tags": ["education"],
        "metadata": {"topic": "quantum computing"}
    }
)
```

## Disable Tracing Temporarily

```python
# Disable for specific call
response = llm.invoke(
    "Secret query",
    config={"callbacks": []}  # Empty callbacks
)

# Or via environment
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

## Team Collaboration

### Create Organization

1. Go to Settings → Organization
2. Invite team members
3. Set roles (Admin, Member, Viewer)

### Shared Projects

```python
# Team members see same traces
os.environ["LANGCHAIN_PROJECT"] = "team-shared-project"
```

## LangSmith Client

### Basic Operations

```python
from langsmith import Client

client = Client()

# List projects
projects = client.list_projects()
for p in projects:
    print(p.name)

# Get runs
runs = client.list_runs(project_name="my-project", limit=10)
for run in runs:
    print(f"{run.name}: {run.status}")

# Get specific run
run = client.read_run(run_id="run-uuid")
print(run.inputs)
print(run.outputs)
```

### Filter Runs

```python
from datetime import datetime, timedelta

# Runs from last 24 hours
yesterday = datetime.now() - timedelta(days=1)

runs = client.list_runs(
    project_name="production",
    start_time=yesterday,
    error=True,  # Only errors
    filter='eq(metadata.user_id, "user-123")'
)
```

## Best Practices

### 1. Project per Environment

```python
# .env.development
LANGCHAIN_PROJECT="app-dev"

# .env.staging
LANGCHAIN_PROJECT="app-staging"

# .env.production
LANGCHAIN_PROJECT="app-production"
```

### 2. Meaningful Run Names

```python
# ❌ Bad
chain.invoke(input)

# ✅ Good
chain.invoke(input, config={"run_name": "user_query_answer"})
```

### 3. Use Tags for Filtering

```python
config = {
    "tags": [
        "production",
        "chatbot",
        f"version-{VERSION}",
        f"model-{MODEL_NAME}"
    ]
}
```

### 4. Include User Context

```python
config = {
    "metadata": {
        "user_id": user.id,
        "session_id": session.id,
        "request_id": request.id,
        "ip_address": request.remote_addr
    }
}
```

## Pricing Notes

- Free tier: Limited traces/month
- Paid: Based on trace volume
- Self-hosted option available

## Ringkasan

1. **LangSmith** - observability platform for LLMs
2. **Environment vars** - simple setup
3. **Automatic tracing** - no code changes needed
4. **Projects** - organize by environment
5. **Metadata/Tags** - enable filtering and analysis

---

**Selanjutnya:** [Tracing & Debugging](/docs/production/tracing) - Analisis traces dan debug issues.
