---
sidebar_position: 4
title: Output Parsers Advanced
description: Parsers untuk situasi khusus dan edge cases
---

# Output Parsers Advanced

Selain `with_structured_output()`, LangChain menyediakan berbagai parsers untuk situasi khusus.

## Kapan Pakai Output Parsers?

| Situasi | Solusi |
|---------|--------|
| Provider modern (OpenAI, Anthropic) | `with_structured_output()` |
| Provider tanpa function calling | `PydanticOutputParser` |
| Streaming JSON | `JsonOutputParser` |
| Simple list extraction | `CommaSeparatedListOutputParser` |
| Fixing malformed output | `OutputFixingParser` |
| Retry on failure | `RetryWithErrorOutputParser` |

## PydanticOutputParser

Menggunakan prompt instructions untuk guide LLM output.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Recipe(BaseModel):
    name: str = Field(description="Recipe name")
    ingredients: list[str] = Field(description="List of ingredients")
    steps: list[str] = Field(description="Cooking steps")
    prep_time: int = Field(description="Preparation time in minutes")

parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = ChatPromptTemplate.from_template("""
Extract recipe information from this text.

{format_instructions}

Text: {text}
""")

chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser

result = chain.invoke(dict(
    text="Pasta Carbonara: Mix eggs, cheese, and bacon. Cook pasta, combine. Takes 20 min prep.",
    format_instructions=parser.get_format_instructions()
))

print(result.name)  # Pasta Carbonara
```

### Format Instructions

Parser automatically generates format instructions:

```python
print(parser.get_format_instructions())
```

Output:
```text
The output should be formatted as a JSON instance that conforms to the JSON schema below.

Here is the output schema:
{"properties": {"name": {"description": "Recipe name", "type": "string"}, ...}}
```

## JsonOutputParser

Untuk JSON output tanpa Pydantic validation.

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

chain = prompt | llm | parser

result = chain.invoke(dict(text="..."))
# Returns: dict
```

### With Pydantic Schema

```python
parser = JsonOutputParser(pydantic_object=Recipe)
# Adds format instructions but returns dict, not Pydantic object
```

### Streaming JSON

JsonOutputParser supports streaming partial JSON:

```python
async for partial in chain.astream(input_data):
    print(partial)  # Partial dict as JSON is built
```

## CommaSeparatedListOutputParser

Simple parser untuk comma-separated lists.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template(
    "List 5 popular programming languages, separated by commas."
)

chain = prompt | llm | parser

result = chain.invoke(dict())
print(result)  # ['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

## OutputFixingParser

Automatically fix malformed output using LLM.

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

base_parser = PydanticOutputParser(pydantic_object=Recipe)

# Wrap with fixing parser
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# If base parser fails, LLM will try to fix the output
result = fixing_parser.parse(malformed_json_string)
```

### How It Works

1. Try to parse with base parser
2. If fails, send error + output to LLM
3. LLM attempts to fix the output
4. Parse fixed output

## RetryWithErrorOutputParser

Retry parsing dengan context error.

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_retries=3
)

# parse_with_prompt includes original prompt for context
result = retry_parser.parse_with_prompt(
    completion=bad_output,
    prompt_value=original_prompt
)
```

## DatetimeOutputParser

Parse dates from natural language.

```python
from langchain.output_parsers import DatetimeOutputParser

parser = DatetimeOutputParser()

prompt = ChatPromptTemplate.from_template(
    "When is {event}? {format_instructions}"
)

chain = prompt | llm | parser

result = chain.invoke(dict(
    event="Christmas 2024",
    format_instructions=parser.get_format_instructions()
))
print(result)  # datetime object: 2024-12-25 00:00:00
```

## EnumOutputParser

Parse to specific enum values.

```python
from langchain.output_parsers import EnumOutputParser
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

parser = EnumOutputParser(enum=Color)

result = parser.parse("The color is red")
print(result)  # Color.RED
```

## Combining Parsers

### Fallback Pattern

```python
from langchain_core.runnables import RunnableWithFallbacks

primary_chain = prompt | llm.with_structured_output(MyModel)
fallback_chain = prompt | llm | PydanticOutputParser(pydantic_object=MyModel)

robust_chain = primary_chain.with_fallbacks([fallback_chain])
```

### Sequential Parsing

```python
from langchain_core.runnables import RunnableLambda

def parse_and_validate(output):
    # First parse
    parsed = json_parser.parse(output)
    # Then validate
    return MyModel(**parsed)

chain = prompt | llm | RunnableLambda(parse_and_validate)
```

## Custom Parser

Buat parser sendiri untuk kebutuhan spesifik.

```python
from langchain_core.output_parsers import BaseOutputParser

class MarkdownTableParser(BaseOutputParser[list[dict]]):
    """Parse markdown table to list of dicts."""
    
    def parse(self, text: str) -> list[dict]:
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return []
        
        # Parse header
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        
        # Parse rows (skip separator line)
        rows = []
        for line in lines[2:]:
            values = [v.strip() for v in line.split('|') if v.strip()]
            if values:
                rows.append(dict(zip(headers, values)))
        
        return rows
    
    @property
    def _type(self) -> str:
        return "markdown_table"

# Usage
parser = MarkdownTableParser()
result = parser.parse("""
| Name | Age |
|------|-----|
| John | 25  |
| Jane | 30  |
""")
print(result)  # [{'Name': 'John', 'Age': '25'}, ...]
```

## Error Handling Strategies

```python
from pydantic import ValidationError

def safe_parse(chain, input_data, default=None):
    try:
        return chain.invoke(input_data)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return default
    except Exception as e:
        print(f"Parse error: {e}")
        return default

result = safe_parse(chain, my_input, default=dict())
```

## Practical Example: Resume Parser

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

class Education(BaseModel):
    degree: str
    institution: str
    year: Optional[int] = None

class Experience(BaseModel):
    title: str
    company: str
    duration: str
    description: str

class Resume(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: str
    skills: List[str]
    education: List[Education]
    experience: List[Experience]

# Create robust parser with fixing capability
base_parser = PydanticOutputParser(pydantic_object=Resume)
llm = ChatOpenAI(model="gpt-4o-mini")

fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm
)

prompt = ChatPromptTemplate.from_template("""
Parse this resume into structured format.

{format_instructions}

Resume:
{resume_text}
""")

chain = prompt | llm | fixing_parser

resume_text = """
JOHN DOE
john.doe@email.com | (555) 123-4567

Software engineer with 5 years of experience in Python and cloud technologies.

SKILLS: Python, AWS, Docker, Kubernetes, PostgreSQL

EDUCATION:
- BS Computer Science, MIT (2018)

EXPERIENCE:
Senior Developer at Tech Corp (2020-Present)
- Led development of microservices architecture
- Improved system performance by 40%

Developer at StartupXYZ (2018-2020)
- Built REST APIs
- Implemented CI/CD pipelines
"""

result = chain.invoke(dict(
    resume_text=resume_text,
    format_instructions=base_parser.get_format_instructions()
))

print(f"Name: {result.name}")
print(f"Skills: {result.skills}")
print(f"Experience: {len(result.experience)} positions")
```

## Ringkasan

1. **PydanticOutputParser** - prompt-based, works everywhere
2. **JsonOutputParser** - streaming JSON support
3. **OutputFixingParser** - auto-fix malformed output
4. **RetryWithErrorOutputParser** - retry with context
5. **Custom parsers** - extend BaseOutputParser
6. Use **fallbacks** untuk robustness

---

## 🎯 Use Case Modul 4: Invoice Extractor API

```python
#!/usr/bin/env python3
"""
Invoice Extractor - Use Case Modul 4
Extract structured data from invoice images/text.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class LineItem(BaseModel):
    description: str = Field(description="Item description")
    quantity: float = Field(description="Number of units")
    unit_price: float = Field(description="Price per unit")
    amount: float = Field(description="Total for this line")

class Invoice(BaseModel):
    """Extracted invoice data."""
    invoice_number: str = Field(description="Invoice ID/number")
    date: str = Field(description="Invoice date (YYYY-MM-DD)")
    vendor: str = Field(description="Seller/vendor name")
    customer: str = Field(description="Buyer/customer name")
    items: List[LineItem] = Field(description="Line items")
    subtotal: float = Field(description="Sum before tax")
    tax: float = Field(description="Tax amount")
    total: float = Field(description="Final total")
    currency: str = Field(default="USD", description="Currency code")
    status: Literal["paid", "unpaid", "partial"] = Field(default="unpaid")

# Setup
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Invoice)

prompt = ChatPromptTemplate.from_template("""
Extract invoice information from this document.
Be precise with numbers and dates.

Document:
{document}
""")

chain = prompt | structured_llm

def extract_invoice(document_text: str) -> Invoice:
    """Extract structured invoice data from text."""
    return chain.invoke(dict(document=document_text))

if __name__ == "__main__":
    sample_invoice = """
    INVOICE #2024-0892
    Date: 2024-12-12
    
    From: ABC Supplies Ltd
    To: XYZ Corporation
    
    Items:
    1. Office Chairs (10) @ $150.00 = $1,500.00
    2. Desks (5) @ $300.00 = $1,500.00
    3. Monitors (10) @ $250.00 = $2,500.00
    
    Subtotal: $5,500.00
    Tax (10%): $550.00
    TOTAL: $6,050.00
    
    Status: Unpaid
    """
    
    result = extract_invoice(sample_invoice)
    
    print("📄 Extracted Invoice Data")
    print("=" * 40)
    print(f"Invoice #: {result.invoice_number}")
    print(f"Date: {result.date}")
    print(f"Vendor: {result.vendor}")
    print(f"Customer: {result.customer}")
    print(f"\nItems:")
    for item in result.items:
        print(f"  - {item.description}: {item.quantity} x ${item.unit_price} = ${item.amount}")
    print(f"\nSubtotal: ${result.subtotal}")
    print(f"Tax: ${result.tax}")
    print(f"Total: ${result.total}")
    print(f"Status: {result.status}")
```

**Selamat!** 🎉 Kamu sudah menyelesaikan Modul 4: Structured Output!

---

**Selanjutnya:** [Modul 5: RAG](/docs/rag/konsep-rag) - Retrieval Augmented Generation untuk knowledge-based applications.
