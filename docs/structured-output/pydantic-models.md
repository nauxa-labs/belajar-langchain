---
sidebar_position: 2
title: Pydantic Models
description: Membuat schema dengan Pydantic untuk structured output
---

# Pydantic Models

Pydantic adalah foundation untuk structured output di LangChain. Di bab ini kita akan belajar cara membuat schema yang efektif.

## Basic Model

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

# Usage
person = Person(name="John", age=30, email="john@example.com")
print(person.name)  # John
print(person.model_dump())  # dict representation
```

## Field Descriptions

Descriptions membantu LLM memahami apa yang diharapkan.

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    """A product listing."""
    
    name: str = Field(description="Product name, max 100 characters")
    price: float = Field(description="Price in USD, must be positive")
    category: str = Field(description="One of: electronics, clothing, food, other")
    in_stock: bool = Field(description="Whether the product is currently available")
```

:::tip
Selalu tambahkan `description` untuk setiap field. Ini seperti "prompt" untuk setiap field.
:::

## Optional Fields

```python
from typing import Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    username: str
    email: str
    bio: Optional[str] = Field(default=None, description="User biography, optional")
    age: Optional[int] = Field(default=None, description="Age in years, optional")
```

## Default Values

```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, gt=0)
    model: str = Field(default="gpt-4o-mini")
```

## Lists dan Nested Models

```python
from typing import List
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    industry: str
    employees: int
    headquarters: Address  # Nested model
    locations: List[Address] = Field(default_factory=list)

# Usage with LLM
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Company)

result = structured_llm.invoke("""
Extract company info:
Apple Inc is a technology company with 150,000 employees.
Headquarters in Cupertino, California, USA.
Also has offices in London, UK and Tokyo, Japan.
""")

print(result.name)  # Apple Inc
print(result.headquarters.city)  # Cupertino
print(len(result.locations))  # 2
```

## Enums untuk Constrained Values

```python
from enum import Enum
from pydantic import BaseModel, Field

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Ticket(BaseModel):
    title: str
    description: str
    sentiment: Sentiment
    priority: Priority
    
# LLM will only output valid enum values
```

## Validators

Custom validation untuk fields.

```python
from pydantic import BaseModel, Field, field_validator

class Email(BaseModel):
    address: str
    subject: str
    body: str
    
    @field_validator('address')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()
    
    @field_validator('subject')
    @classmethod
    def validate_subject(cls, v):
        if len(v) > 100:
            raise ValueError('Subject too long')
        return v
```

## Literal Types

Untuk nilai yang sangat spesifik.

```python
from typing import Literal
from pydantic import BaseModel

class Classification(BaseModel):
    category: Literal["spam", "not_spam"]
    confidence: float

class Action(BaseModel):
    action_type: Literal["create", "update", "delete"]
    target: str
```

## Complex Example: Invoice Extractor

```python
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import date

class LineItem(BaseModel):
    """A single line item on an invoice."""
    description: str = Field(description="Item description")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit in USD")
    total: float = Field(description="Line total (quantity * unit_price)")

class Invoice(BaseModel):
    """An invoice document."""
    
    invoice_number: str = Field(description="Unique invoice identifier")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    
    vendor_name: str = Field(description="Name of the vendor/seller")
    vendor_address: Optional[str] = Field(default=None)
    
    customer_name: str = Field(description="Name of the customer/buyer")
    customer_address: Optional[str] = Field(default=None)
    
    line_items: List[LineItem] = Field(description="List of items/services")
    
    subtotal: float = Field(description="Sum of all line items")
    tax_rate: float = Field(description="Tax rate as decimal (e.g., 0.1 for 10%)")
    tax_amount: float = Field(description="Tax amount in USD")
    total: float = Field(description="Final total including tax")
    
    payment_terms: Optional[str] = Field(default=None, description="e.g., Net 30")
    status: Literal["paid", "unpaid", "overdue"] = Field(default="unpaid")

# Use with LLM
structured_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Invoice)

invoice_text = """
INVOICE #INV-2024-001
Date: December 12, 2024

From: Tech Solutions Inc
123 Main St, San Francisco, CA

To: Acme Corp
456 Oak Ave, New York, NY

Items:
- Web Development Services (40 hours) @ $150/hr = $6,000
- Cloud Hosting (1 month) @ $200/mo = $200
- Domain Registration (1 year) @ $15 = $15

Subtotal: $6,215
Tax (10%): $621.50
Total: $6,836.50

Payment Terms: Net 30
Status: Unpaid
"""

invoice = structured_llm.invoke(f"Extract invoice data:\n{invoice_text}")
print(f"Invoice: {invoice.invoice_number}")
print(f"Total: ${invoice.total}")
print(f"Items: {len(invoice.line_items)}")
```

## Model Config

Customize model behavior.

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,  # Strict type checking
        extra='forbid',  # No extra fields allowed
        frozen=True,  # Immutable
    )
    
    name: str
    value: int
```

## Best Practices

### 1. Descriptive Field Names

```python
# ❌ Bad
class Data(BaseModel):
    n: str
    v: int

# ✅ Good
class Product(BaseModel):
    product_name: str
    quantity: int
```

### 2. Always Add Descriptions

```python
# ❌ Bad
class Task(BaseModel):
    title: str
    done: bool

# ✅ Good
class Task(BaseModel):
    title: str = Field(description="Short task title, max 100 chars")
    done: bool = Field(description="Whether task is completed")
```

### 3. Use Appropriate Types

```python
# ❌ Bad - age as string
class Person(BaseModel):
    age: str  # "25"

# ✅ Good - age as int
class Person(BaseModel):
    age: int  # 25
```

### 4. Constrain Values

```python
from pydantic import Field

class Rating(BaseModel):
    # ❌ Bad - no constraints
    score: int
    
    # ✅ Good - with constraints
    score: int = Field(ge=1, le=5, description="Rating from 1-5")
```

## Ringkasan

1. **BaseModel** = foundation untuk schema
2. **Field()** = add descriptions dan constraints
3. **Optional** = nullable fields
4. **List** = arrays of items
5. **Nested models** = complex structures
6. **Enum** = constrained choices
7. **Literal** = exact values
8. **Validators** = custom validation

---

**Selanjutnya:** [with_structured_output](/docs/structured-output/with-structured-output) - Method utama untuk structured output.
