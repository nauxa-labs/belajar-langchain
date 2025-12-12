---
sidebar_position: 1
title: Mengapa Structured Output
description: Pentingnya output terstruktur dan cara LangChain menanganinya
---

# Mengapa Structured Output?

LLM secara default menghasilkan teks bebas. Tapi untuk aplikasi real-world, kita sering butuh data dalam format yang **terstruktur dan predictable**.

## Masalah dengan Unstructured Output

```python
# Unstructured - susah di-parse
response = llm.invoke("Extract name and age from: John is 25 years old")
# Output: "The name is John and the age is 25."
# Atau: "Name: John, Age: 25"
# Atau: "John, 25"
```

Masalah:
- **Inconsistent format** - tidak bisa diprediksi
- **Hard to parse** - perlu regex atau string manipulation
- **Error prone** - parsing bisa gagal
- **Not type-safe** - tidak ada validasi

## Solusi: Structured Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Structured - selalu sama formatnya
result = llm.with_structured_output(Person).invoke(
    "Extract name and age from: John is 25 years old"
)
# Output: Person(name='John', age=25)

print(result.name)  # "John"
print(result.age)   # 25 (integer, bukan string!)
```

## Use Cases

| Use Case | Why Structured? |
|----------|-----------------|
| Data extraction | Parse ke database fields |
| Form filling | Validasi input |
| API responses | Consistent JSON schema |
| Tool calling | Parameters harus typed |
| Classification | Enumerate valid options |
| Entity recognition | Extract specific fields |

## Cara LangChain Handle Structured Output

### 1. with_structured_output() (Recommended)

Menggunakan function calling atau JSON mode dari provider.

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    genre: str

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Movie)

result = structured_llm.invoke("The movie Inception was released in 2010, it's a sci-fi thriller")
print(result)  # Movie(title='Inception', year=2010, genre='sci-fi thriller')
```

### 2. PydanticOutputParser

Menggunakan prompt instructions untuk format output.

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Movie)
# Menambahkan format instructions ke prompt
```

### 3. JsonOutputParser

Untuk output JSON tanpa Pydantic validation.

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
```

## Provider Support

| Provider | Function Calling | JSON Mode |
|----------|-----------------|-----------|
| OpenAI | ✅ | ✅ |
| Anthropic | ✅ | ✅ |
| Google | ✅ | ✅ |
| Ollama | ⚠️ Some models | ⚠️ Some models |
| Groq | ✅ | ✅ |

## Pydantic Basics

Pydantic adalah library untuk data validation di Python.

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    """A task item with details."""
    
    title: str = Field(description="Short title of the task")
    description: Optional[str] = Field(default=None, description="Detailed description")
    priority: Priority = Field(default=Priority.MEDIUM)
    tags: List[str] = Field(default_factory=list)
    completed: bool = False

# Validation happens automatically
task = Task(title="Learn LangChain", priority="high", tags=["ai", "python"])
print(task.model_dump())
```

## Keuntungan Structured Output

1. **Type Safety** - Validasi otomatis
2. **IDE Support** - Autocomplete dan type hints
3. **Consistent** - Format selalu sama
4. **Maintainable** - Schema sebagai documentation
5. **Integration Ready** - Langsung bisa masuk database/API

## Ringkasan

- **Unstructured output** = teks bebas, susah di-parse
- **Structured output** = format pasti, type-safe
- **with_structured_output()** = cara paling mudah
- **Pydantic** = foundation untuk schema definition
- Pilih method sesuai **provider support**

---

**Selanjutnya:** [Pydantic Models](/docs/structured-output/pydantic-models) - Membuat schema dengan Pydantic.
