---
sidebar_position: 5
title: Error Handling
description: Retry, fallbacks, dan error recovery dalam LCEL chains
---

# Error Handling dalam LCEL

Aplikasi production perlu menangani errors dengan baik. LCEL menyediakan built-in mechanisms untuk retry, fallbacks, dan error recovery.

## Common Errors

| Error Type | Cause | Solution |
|------------|-------|----------|
| `RateLimitError` | Terlalu banyak requests | Retry dengan backoff |
| `APIError` | Provider API down | Fallback ke provider lain |
| `TimeoutError` | Request terlalu lama | Retry atau timeout handling |
| `OutputParserException` | LLM output tidak sesuai format | Retry atau fixing parser |
| `ValidationError` | Input tidak valid | Input validation |

## `with_retry()` - Automatic Retry

Retry otomatis dengan exponential backoff.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Add retry to LLM
llm_with_retry = llm.with_retry(
    stop_after_attempt=3,  # Max 3 attempts
    wait_exponential_jitter=True  # Add randomness to prevent thundering herd
)

chain = (
    ChatPromptTemplate.from_template("Explain {topic}")
    | llm_with_retry
    | StrOutputParser()
)

# Will automatically retry on transient errors
result = chain.invoke({"topic": "quantum computing"})
```

### Retry Configuration

```python
from tenacity import stop_after_attempt, wait_exponential

llm_with_retry = llm.with_retry(
    retry_if_exception_type=(RateLimitError, APIError),
    stop_after_attempt=5,
    wait_exponential_multiplier=1,
    wait_exponential_max=60
)
```

### Retry pada Specific Exceptions

```python
from openai import RateLimitError, APIError

llm_with_retry = llm.with_retry(
    retry_if_exception_type=(RateLimitError,),  # Only retry rate limits
    stop_after_attempt=3
)
```

## `with_fallbacks()` - Backup Chains

Jika chain utama gagal, gunakan fallback.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Primary model
primary_llm = ChatOpenAI(model="gpt-4o")

# Fallback models
fallback_llm_1 = ChatOpenAI(model="gpt-4o-mini")
fallback_llm_2 = ChatAnthropic(model="claude-3-haiku-20240307")

# LLM with fallbacks
robust_llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])

# Use in chain
chain = prompt | robust_llm | parser
```

### Fallback Order

```python
# Fallback order: gpt-4 → gpt-3.5 → claude
# Jika gpt-4 gagal → coba gpt-3.5
# Jika gpt-3.5 gagal → coba claude
# Jika semuanya gagal → raise exception

robust_llm = (
    ChatOpenAI(model="gpt-4")
    .with_fallbacks([
        ChatOpenAI(model="gpt-3.5-turbo"),
        ChatAnthropic(model="claude-3-haiku-20240307")
    ])
)
```

### Fallback untuk Entire Chains

```python
main_chain = prompt | expensive_llm | parser
fallback_chain = prompt | cheap_llm | parser

robust_chain = main_chain.with_fallbacks([fallback_chain])
```

## Exception Handling dengan Try/Except

Manual error handling dalam chain.

```python
from langchain_core.runnables import RunnableLambda

def safe_invoke(chain, input_data, default="Sorry, I couldn't process that."):
    """Safely invoke chain with default on error."""
    try:
        return chain.invoke(input_data)
    except Exception as e:
        print(f"Error: {e}")
        return default

# As Runnable
def safe_wrapper(data: dict) -> str:
    try:
        return expensive_chain.invoke(data)
    except Exception:
        return cheap_chain.invoke(data)

safe_runnable = RunnableLambda(safe_wrapper)
```

## Combining Retry dan Fallback

```python
from langchain_openai import ChatOpenAI

# Primary with retry
primary = ChatOpenAI(model="gpt-4o").with_retry(
    stop_after_attempt=2,
    wait_exponential_jitter=True
)

# Fallback with retry
fallback = ChatOpenAI(model="gpt-4o-mini").with_retry(
    stop_after_attempt=3
)

# Combined: try primary (with retries), then fallback (with retries)
robust_llm = primary.with_fallbacks([fallback])
```

## Timeout Handling

```python
from langchain_openai import ChatOpenAI

# Set timeout
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30.0,  # 30 seconds
    max_retries=2
)

# Per-request timeout
result = chain.invoke(
    {"topic": "AI"},
    config={"timeout": 10}  # 10 seconds for this request
)
```

## Output Parser Error Recovery

### OutputFixingParser

Otomatis fix parsing errors.

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

base_parser = PydanticOutputParser(pydantic_object=Person)

# Wrap dengan fixing parser
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# Jika parsing gagal, LLM akan mencoba fix
result = fixing_parser.parse(malformed_output)
```

### RetryWithErrorOutputParser

Retry parsing dengan context error.

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_retries=2
)

result = retry_parser.parse_with_prompt(
    completion=bad_output,
    prompt_value=original_prompt
)
```

## Graceful Degradation Pattern

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

def get_with_degradation(primary_func, fallback_value):
    """Return fallback value if primary fails."""
    def wrapper(data):
        try:
            return primary_func(data)
        except Exception as e:
            print(f"Degrading due to: {e}")
            return fallback_value
    return RunnableLambda(wrapper)

# Use for optional enrichment
enrich_chain = RunnableParallel(
    main_content=main_chain,  # Required
    extra_info=get_with_degradation(extra_chain, "Not available"),  # Optional
    recommendations=get_with_degradation(rec_chain, [])  # Optional
)
```

## Logging Errors

```python
from langchain_core.callbacks import BaseCallbackHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorLoggingHandler(BaseCallbackHandler):
    def on_chain_error(self, error, **kwargs):
        logger.error(f"Chain error: {error}")
    
    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM error: {error}")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"Tool error: {error}")

# Use
result = chain.invoke(
    input_data,
    config={"callbacks": [ErrorLoggingHandler()]}
)
```

## Practical Example: Robust API

```python
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Build robust chain
primary_llm = ChatOpenAI(model="gpt-4o").with_retry(stop_after_attempt=2)
fallback_llm = ChatAnthropic(model="claude-3-haiku-20240307").with_retry(stop_after_attempt=2)

robust_chain = (
    ChatPromptTemplate.from_template("Answer: {question}")
    | primary_llm.with_fallbacks([fallback_llm])
    | StrOutputParser()
)

@app.get("/ask")
async def ask(question: str):
    try:
        result = await robust_chain.ainvoke({"question": question})
        return {"answer": result, "status": "success"}
    except Exception as e:
        logger.error(f"All models failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )
```

## Error Handling Checklist

✅ **Add retry** untuk transient errors (rate limits, timeouts)
✅ **Add fallbacks** untuk critical paths
✅ **Set timeouts** untuk semua external calls
✅ **Log errors** untuk debugging
✅ **Graceful degradation** untuk non-critical features
✅ **User-friendly error messages** di production

## Ringkasan

1. **`with_retry()`** - automatic retry dengan exponential backoff
2. **`with_fallbacks()`** - backup chains jika primary gagal
3. **Combine** retry + fallback untuk maximum robustness
4. **OutputFixingParser** - auto-fix parsing errors
5. **Graceful degradation** - fallback values untuk optional features
6. **Logging** - track errors untuk debugging

---

**Selanjutnya:** [Streaming](/docs/lcel/streaming) - Deep dive ke streaming output dan events.
