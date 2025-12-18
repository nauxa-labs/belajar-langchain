---
sidebar_position: 5
title: Best Practices
description: Production-ready patterns untuk LLM applications
---

# Production Best Practices

Patterns dan strategies untuk **robust, scalable** LLM applications.

## Caching

### LLM Response Caching

```python
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Now identical prompts return cached responses
llm.invoke("What is Python?")  # First call: hits API
llm.invoke("What is Python?")  # Second call: from cache
```

### Redis Cache (Production)

```python
from langchain_community.cache import RedisCache
import redis

redis_client = redis.Redis.from_url("redis://localhost:6379")
set_llm_cache(RedisCache(redis_client))
```

### Semantic Cache

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95  # Similarity threshold
))

# Similar prompts also hit cache
llm.invoke("What is Python?")
llm.invoke("Can you explain Python?")  # Semantic match!
```

## Rate Limiting

### Client-Side

```python
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_openai import ChatOpenAI

@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5)
)
def call_llm(prompt: str):
    return llm.invoke(prompt)
```

### Token Bucket

```python
import time
from threading import Lock

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.calls_per_minute,
                self.tokens + elapsed * (self.calls_per_minute / 60)
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    def wait_and_acquire(self):
        while not self.acquire():
            time.sleep(0.1)

limiter = RateLimiter(calls_per_minute=60)

def rate_limited_call(prompt: str):
    limiter.wait_and_acquire()
    return llm.invoke(prompt)
```

## Error Handling

### Retry with Fallback

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary_llm = ChatOpenAI(model="gpt-4o-mini")
fallback_llm = ChatAnthropic(model="claude-3-haiku-20240307")

chain_with_fallback = primary_llm.with_fallbacks([fallback_llm])

# If OpenAI fails, tries Claude
response = chain_with_fallback.invoke("Hello!")
```

### Graceful Degradation

```python
async def safe_generate(prompt: str, context: str = None) -> str:
    """Generate with multiple fallback levels."""
    
    # Level 1: Full RAG
    if context:
        try:
            return await rag_chain.ainvoke({
                "question": prompt,
                "context": context
            })
        except Exception as e:
            logging.warning(f"RAG failed: {e}")
    
    # Level 2: Direct LLM
    try:
        return await llm.ainvoke(prompt)
    except Exception as e:
        logging.warning(f"LLM failed: {e}")
    
    # Level 3: Cached/static response
    cached = get_cached_response(prompt)
    if cached:
        return cached
    
    # Level 4: Error message
    return "I'm sorry, I'm having trouble right now. Please try again later."
```

### Circuit Breaker

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def reset(self):
        self.failures = 0
        self.state = "closed"

breaker = CircuitBreaker()

def safe_llm_call(prompt):
    return breaker.call(llm.invoke, prompt)
```

## Monitoring & Alerting

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
llm_requests = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])
llm_latency = Histogram('llm_latency_seconds', 'LLM latency', ['model'])
token_usage = Counter('llm_tokens_total', 'Token usage', ['model', 'type'])

def monitored_invoke(llm, prompt):
    model = llm.model_name
    
    with llm_latency.labels(model=model).time():
        try:
            response = llm.invoke(prompt)
            llm_requests.labels(model=model, status='success').inc()
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                token_usage.labels(model=model, type='input').inc(
                    response.usage_metadata.get('input_tokens', 0)
                )
                token_usage.labels(model=model, type='output').inc(
                    response.usage_metadata.get('output_tokens', 0)
                )
            
            return response
        except Exception as e:
            llm_requests.labels(model=model, status='error').inc()
            raise

# Start metrics server
start_http_server(9090)  # Prometheus scrapes this
```

### Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log(self, event: str, **kwargs):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            **kwargs
        }
        self.logger.info(json.dumps(record))

logger = StructuredLogger("llm-app")

def logged_invoke(prompt: str):
    start = time.time()
    logger.log("llm_request_started", prompt_length=len(prompt))
    
    try:
        response = llm.invoke(prompt)
        latency = time.time() - start
        
        logger.log(
            "llm_request_completed",
            latency_ms=latency * 1000,
            output_length=len(response.content)
        )
        return response
    except Exception as e:
        logger.log("llm_request_failed", error=str(e))
        raise
```

## Cost Control

### Budget Limits

```python
import os
from threading import Lock

class BudgetTracker:
    def __init__(self, daily_budget: float):
        self.daily_budget = daily_budget
        self.spent_today = 0.0
        self.lock = Lock()
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        # Pricing per 1M tokens (example for GPT-4o-mini)
        prices = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00}
        }
        
        p = prices.get(model, {"input": 1.0, "output": 3.0})
        return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    
    def can_proceed(self, estimated_cost: float) -> bool:
        with self.lock:
            return self.spent_today + estimated_cost <= self.daily_budget
    
    def record_spend(self, cost: float):
        with self.lock:
            self.spent_today += cost

budget = BudgetTracker(daily_budget=10.0)  # $10/day

def budget_controlled_call(prompt: str):
    # Estimate cost (rough: 1 token ≈ 4 chars)
    estimated_tokens = len(prompt) / 4 + 500  # Input + expected output
    estimated_cost = budget.estimate_cost(
        int(estimated_tokens * 0.3),
        int(estimated_tokens * 0.7),
        "gpt-4o-mini"
    )
    
    if not budget.can_proceed(estimated_cost):
        raise Exception("Daily budget exceeded")
    
    response = llm.invoke(prompt)
    
    # Record actual cost
    if hasattr(response, 'usage_metadata'):
        actual_cost = budget.estimate_cost(
            response.usage_metadata.get('input_tokens', 0),
            response.usage_metadata.get('output_tokens', 0),
            "gpt-4o-mini"
        )
        budget.record_spend(actual_cost)
    
    return response
```

## Security

### Input Sanitization

```python
import re

def sanitize_input(text: str) -> str:
    """Remove potential prompt injection patterns."""
    
    # Remove common injection patterns
    patterns = [
        r"ignore previous instructions",
        r"disregard above",
        r"forget everything",
        r"new instructions:",
        r"system:",
    ]
    
    sanitized = text.lower()
    for pattern in patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValueError("Potential prompt injection detected")
    
    # Limit length
    if len(text) > 10000:
        text = text[:10000]
    
    return text
```

### Output Validation

```python
def validate_output(response: str, expected_format: str = None) -> str:
    """Validate LLM output before returning."""
    
    # Check for sensitive data leakage
    sensitive_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"sk-[a-zA-Z0-9]{48}",  # API keys
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, response):
            return "[REDACTED - Sensitive information detected]"
    
    return response
```

## Use Case: Document Q&A API

```python
#!/usr/bin/env python3
"""
Production Document Q&A API
"""

from fastapi import FastAPI, HTTPException, Depends
from langserve import add_routes
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.cache import RedisCache
from langchain.globals import set_llm_cache
import redis
import os

# Setup
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
set_llm_cache(RedisCache(redis_client))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context. If unsure, say so.\n\nContext: {context}"),
    ("human", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# App
app = FastAPI(title="Document Q&A API")

add_routes(app, rag_chain, path="/qa")

@app.get("/health")
async def health():
    # Check dependencies
    try:
        redis_client.ping()
        return {"status": "healthy"}
    except:
        raise HTTPException(503, "Service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Ringkasan

1. **Caching** - reduce API calls & costs
2. **Rate limiting** - prevent quota issues
3. **Error handling** - fallbacks & circuit breakers
4. **Monitoring** - metrics & structured logging
5. **Cost control** - budgets & tracking
6. **Security** - sanitize inputs, validate outputs

---

**Selamat!** 🎉 Kamu sudah menyelesaikan **Modul 9: Production & Observability**!

Kamu sekarang bisa:
- Setup LangSmith monitoring
- Debug dengan tracing
- Evaluate LLM outputs
- Deploy dengan LangServe
- Implement production patterns

---

**Selanjutnya:** [Modul 10: Proyek Praktis](/docs/projects/intro) - Apply everything in real projects.
