---
sidebar_position: 2
title: Tracing & Debugging
description: Analisis traces dan debug LLM issues
---

# Tracing & Debugging

Memahami **apa yang terjadi** di dalam LLM application dan **debug issues** dengan efektif.

## Understanding Traces

### Trace Structure

```
Trace (Run)
├── Span: Chain
│   ├── Span: Prompt Template
│   ├── Span: LLM Call
│   │   ├── Input tokens
│   │   ├── Output tokens
│   │   └── Latency
│   └── Span: Output Parser
└── Metadata & Tags
```

### Viewing in LangSmith

1. Go to project dashboard
2. Click on a run
3. Expand spans to see details
4. View inputs/outputs at each step

## Latency Analysis

### Identify Slow Steps

```python
from langsmith import Client

client = Client()

# Get runs with high latency
runs = client.list_runs(
    project_name="production",
    filter="gt(latency, 5)"  # > 5 seconds
)

for run in runs:
    print(f"{run.name}: {run.total_time:.2f}s")
```

### Common Latency Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow LLM | Large prompt/response | Reduce context, use smaller model |
| Slow retrieval | Inefficient search | Optimize vector store, add caching |
| Network | API distance | Use regional endpoints |
| Cold start | New instances | Keep-alive, pre-warming |

## Token Usage & Cost Tracking

### View in Dashboard

LangSmith automatically tracks:
- Input tokens
- Output tokens
- Total tokens
- Estimated cost

### Programmatic Access

```python
from langsmith import Client

client = Client()

runs = client.list_runs(
    project_name="production",
    start_time=yesterday
)

total_tokens = 0
total_cost = 0

for run in runs:
    if run.token_usage:
        total_tokens += run.token_usage.get("total_tokens", 0)
        # Estimate cost (example for GPT-4o-mini)
        input_cost = run.token_usage.get("input_tokens", 0) * 0.00015 / 1000
        output_cost = run.token_usage.get("output_tokens", 0) * 0.0006 / 1000
        total_cost += input_cost + output_cost

print(f"Total tokens: {total_tokens:,}")
print(f"Estimated cost: ${total_cost:.2f}")
```

## Error Debugging

### Find Errors

```python
# Get failed runs
failed_runs = client.list_runs(
    project_name="production",
    error=True,
    limit=50
)

for run in failed_runs:
    print(f"Error: {run.error}")
    print(f"Inputs: {run.inputs}")
    print("---")
```

### Common Errors

#### Rate Limiting

```python
# Error: openai.RateLimitError

# Solution: Add retry logic
from langchain_openai import ChatOpenAI
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(min=1, max=60))
def call_with_retry(llm, prompt):
    return llm.invoke(prompt)
```

#### Context Length Exceeded

```python
# Error: context_length_exceeded

# Solution: Trim messages
from langchain_core.messages import trim_messages

messages = trim_messages(
    messages,
    max_tokens=4000,
    token_counter=len,
    strategy="last"
)
```

#### Timeout

```python
# Error: timeout

# Solution: Increase timeout
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=60,  # seconds
    max_retries=3
)
```

## Debugging Workflow

### 1. Find Problematic Run

In LangSmith:
- Filter by error status
- Filter by high latency
- Filter by user feedback

### 2. Inspect the Trace

```
Click run → View spans → Check each step:
- Were inputs correct?
- Was the prompt well-formed?
- Was the LLM response appropriate?
- Did parsing succeed?
```

### 3. Reproduce Locally

```python
# Get run details
run = client.read_run(run_id="problematic-run-id")

# Get inputs
original_input = run.inputs

# Reproduce
result = chain.invoke(original_input)
```

### 4. Add Debugging Logs

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# LangChain will log details
from langchain.globals import set_debug
set_debug(True)
```

## Custom Debugging

### Add Checkpoints

```python
from langchain_core.runnables import RunnableLambda

def debug_step(data):
    print(f"DEBUG: {type(data)} - {str(data)[:100]}")
    return data

chain = (
    prompt 
    | RunnableLambda(debug_step)  # Debug after prompt
    | llm 
    | RunnableLambda(debug_step)  # Debug after LLM
    | parser
)
```

### Capture Intermediate States

```python
from langchain_core.tracers import ConsoleCallbackHandler

# Print all steps to console
handler = ConsoleCallbackHandler()

response = chain.invoke(
    {"input": "test"},
    config={"callbacks": [handler]}
)
```

## Comparing Runs

### A/B Testing

```python
# Version A
response_a = chain.invoke(
    input,
    config={"tags": ["version-a"], "run_name": "test-v1"}
)

# Version B (different prompt)
response_b = chain_v2.invoke(
    input,
    config={"tags": ["version-b"], "run_name": "test-v2"}
)
```

In LangSmith:
1. Filter by tag "version-a"
2. Compare metrics with "version-b"

### Regression Detection

```python
# Compare this week vs last week
from datetime import datetime, timedelta

this_week = client.list_runs(
    project_name="production",
    start_time=datetime.now() - timedelta(days=7)
)

last_week = client.list_runs(
    project_name="production",
    start_time=datetime.now() - timedelta(days=14),
    end_time=datetime.now() - timedelta(days=7)
)

# Compare latency
avg_latency_this_week = sum(r.total_time for r in this_week) / len(this_week)
avg_latency_last_week = sum(r.total_time for r in last_week) / len(last_week)

if avg_latency_this_week > avg_latency_last_week * 1.2:
    print("⚠️ Latency increased by 20%!")
```

## Dashboard Tips

### Useful Filters

```
# High latency production runs
project:production AND gt(latency, 3)

# Errors from specific user
error:true AND eq(metadata.user_id, "123")

# Specific tag
tags:chatbot AND tags:v2
```

### Alerts (Paid Feature)

Set up alerts for:
- Error rate > threshold
- Latency > threshold
- Cost > budget

## Ringkasan

1. **Traces** show full execution path
2. **Latency** analysis per span
3. **Token/cost** tracking automatic
4. **Errors** searchable and filterable
5. **Reproduction** from saved inputs
6. **Comparison** for A/B and regression

---

**Selanjutnya:** [Evaluation](/docs/production/evaluation) - Test dan evaluate LLM outputs.
