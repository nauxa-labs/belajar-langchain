---
sidebar_position: 3
title: with_structured_output
description: Method utama untuk menghasilkan structured output dari LLM
---

# with_structured_output()

Method `with_structured_output()` adalah cara paling mudah dan reliable untuk mendapatkan structured output dari LLM.

## Basic Usage

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Joke(BaseModel):
    """A joke with setup and punchline."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a programming joke")
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")
```

## Cara Kerjanya

`with_structured_output()` menggunakan fitur native dari provider:

1. **Function Calling** (OpenAI, Anthropic) - Model dipaksa memanggil "function" dengan schema tertentu
2. **JSON Mode** - Model dipaksa output valid JSON

```python
# Behind the scenes, LangChain:
# 1. Converts Pydantic model to JSON schema
# 2. Sends schema to LLM via function calling
# 3. Parses response back to Pydantic object
```

## Method Options

### Default Mode

```python
# Uses function calling by default (most reliable)
structured_llm = llm.with_structured_output(MyModel)
```

### JSON Mode

```python
# Force JSON mode instead of function calling
structured_llm = llm.with_structured_output(MyModel, method="json_mode")
```

### Include Raw Response

```python
# Get both parsed result and raw LLM response
structured_llm = llm.with_structured_output(MyModel, include_raw=True)

response = structured_llm.invoke("...")
print(response["parsed"])  # Pydantic object
print(response["raw"])     # Raw AIMessage
```

## With Different Providers

### OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
structured = llm.with_structured_output(MyModel)
```

### Anthropic

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307")
structured = llm.with_structured_output(MyModel)
```

### Google

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
structured = llm.with_structured_output(MyModel)
```

### Ollama

```python
from langchain_ollama import ChatOllama

# Note: Not all Ollama models support structured output
llm = ChatOllama(model="llama3.1")
structured = llm.with_structured_output(MyModel)
```

## In LCEL Chains

```python
from langchain_core.prompts import ChatPromptTemplate

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    key_points: list[str]

prompt = ChatPromptTemplate.from_template(
    "Analyze this text: {text}"
)

chain = prompt | llm.with_structured_output(Analysis)

result = chain.invoke(dict(text="I love this product! It works great."))
print(result.sentiment)  # positive
print(result.key_points)  # ['loves product', 'works great']
```

## Multiple Output Types

Untuk cases dengan multiple possible output types.

```python
from typing import Union

class SuccessResponse(BaseModel):
    success: bool = True
    data: str

class ErrorResponse(BaseModel):
    success: bool = False
    error_message: str
    error_code: int

# Union type
ResponseType = Union[SuccessResponse, ErrorResponse]

structured_llm = llm.with_structured_output(ResponseType)
```

## Streaming Structured Output

Partial objects selama streaming.

```python
async def stream_structured():
    async for partial in structured_llm.astream("Tell me about Python"):
        print(partial)
        # Prints partial objects as they're built
```

## Error Handling

```python
from pydantic import ValidationError

try:
    result = structured_llm.invoke("Invalid input that might fail")
except ValidationError as e:
    print(f"Validation failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

## Practical Examples

### Entity Extraction

```python
from typing import List
from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Type: person, organization, location, date")
    context: str = Field(description="How entity appears in text")

class Entities(BaseModel):
    entities: List[Entity]

extractor = llm.with_structured_output(Entities)

text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino on September 12, 2023."
result = extractor.invoke(f"Extract entities from: {text}")

for entity in result.entities:
    print(f"{entity.entity_type}: {entity.name}")
```

### Classification

```python
from typing import Literal

class Classification(BaseModel):
    category: Literal["bug", "feature", "question", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str

classifier = llm.with_structured_output(Classification)

ticket = "The app crashes when I try to upload files larger than 10MB"
result = classifier.invoke(f"Classify this support ticket: {ticket}")

print(f"Category: {result.category}")  # bug
print(f"Urgency: {result.urgency}")    # high
```

### Data Transformation

```python
class StructuredData(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    company: str

transformer = llm.with_structured_output(StructuredData)

messy_input = """
Contact: john.doe@acme.com
John Doe from Acme Corp
Phone: (555) 123-4567
"""

result = transformer.invoke(f"Extract contact info: {messy_input}")
print(result.model_dump())
```

## Comparison with Other Methods

| Method | Reliability | Streaming | Provider Support |
|--------|-------------|-----------|------------------|
| with_structured_output() | High | Yes | Most providers |
| PydanticOutputParser | Medium | Limited | All (prompt-based) |
| JsonOutputParser | Medium | Yes | All |

## Best Practices

### 1. Use Descriptive Schemas

```python
# ❌ Bad
class X(BaseModel):
    a: str
    b: int

# ✅ Good
class Product(BaseModel):
    """Product information extracted from description."""
    name: str = Field(description="Product name")
    price: int = Field(description="Price in cents")
```

### 2. Handle Failures

```python
from langchain_core.runnables import RunnableWithFallbacks

# Add fallback for robustness
structured = llm.with_structured_output(MyModel)
safe_structured = structured.with_fallbacks([
    llm.with_structured_output(MyModel, method="json_mode")
])
```

### 3. Validate Critical Fields

```python
class StrictModel(BaseModel):
    required_field: str = Field(min_length=1)
    numeric_field: int = Field(ge=0, le=100)
```

## Ringkasan

1. **with_structured_output()** = method utama untuk structured output
2. Uses **function calling** for reliability
3. Returns **Pydantic objects** langsung
4. Works with **LCEL chains**
5. Supports **streaming** partial objects
6. Available on most **modern LLM providers**

---

**Selanjutnya:** [Output Parsers Deep Dive](/docs/structured-output/output-parsers-advanced) - Parsers untuk situasi khusus.
