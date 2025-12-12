---
sidebar_position: 4
title: Output Parsers
description: Memproses output LLM menjadi format yang terstruktur
---

# Output Parsers

Output Parsers memungkinkan kita mengubah output LLM yang berupa free-text menjadi format terstruktur seperti JSON, list, atau custom objects.

## Mengapa Perlu Output Parsers?

```python
# Tanpa parser - output adalah string mentah
response = llm.invoke("List 3 programming languages")
print(response.content)
# "Here are 3 programming languages:\n1. Python\n2. JavaScript\n3. Go"

# Sulit diproses lebih lanjut!
```

Dengan Output Parsers, kita bisa mendapatkan data yang structured dan siap diproses.

## StrOutputParser - String Output

Parser paling sederhana - extract string dari AIMessage.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = ChatPromptTemplate.from_template("Jelaskan {topik} dalam 1 kalimat")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Chain
chain = template | llm | parser

result = chain.invoke({"topik": "machine learning"})
print(type(result))  # <class 'str'>
print(result)
# "Machine learning adalah cabang AI yang memungkinkan komputer belajar dari data."
```

## JsonOutputParser - JSON Output

Untuk output berupa JSON/dictionary.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Template dengan instruksi JSON
template = ChatPromptTemplate.from_template("""
Extract informasi dari teks berikut ke format JSON:
- nama: nama orang
- umur: umur dalam angka  
- pekerjaan: pekerjaan

Teks: {teks}

Respond with valid JSON only.
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

chain = template | llm | parser

result = chain.invoke({
    "teks": "Budi adalah seorang programmer berusia 28 tahun yang tinggal di Jakarta."
})

print(type(result))  # <class 'dict'>
print(result)
# {'nama': 'Budi', 'umur': 28, 'pekerjaan': 'programmer'}

# Akses seperti dictionary biasa
print(result["nama"])  # Budi
```

### Streaming JSON

```python
# JsonOutputParser juga support streaming!
async for chunk in chain.astream({"teks": "..."}):
    print(chunk)  # Partial JSON dict
```

## CommaSeparatedListOutputParser

Untuk output berupa list.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

template = ChatPromptTemplate.from_template(
    "List 5 {kategori}. Output as comma-separated values only."
)

chain = template | ChatOpenAI(model="gpt-4o-mini") | parser

result = chain.invoke({"kategori": "warna"})
print(type(result))  # <class 'list'>
print(result)
# ['merah', 'biru', 'hijau', 'kuning', 'ungu']
```

## PydanticOutputParser - Typed & Validated

Parser paling powerful - output langsung menjadi Pydantic model dengan validation.

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Define Pydantic model
class Person(BaseModel):
    """Informasi tentang seseorang."""
    nama: str = Field(description="Nama lengkap orang")
    umur: int = Field(description="Umur dalam tahun")
    pekerjaan: str = Field(description="Pekerjaan atau profesi")
    skills: list[str] = Field(description="List skill yang dimiliki")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Template dengan format instructions
template = ChatPromptTemplate.from_template("""
Extract informasi dari teks berikut.

{format_instructions}

Teks: {teks}
""")

# Inject format instructions
prompt = template.partial(format_instructions=parser.get_format_instructions())

# Chain
chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser

result = chain.invoke({
    "teks": "Andi adalah data scientist berusia 30 tahun. Dia ahli Python, SQL, dan Machine Learning."
})

print(type(result))  # <class '__main__.Person'>
print(result)
# nama='Andi' umur=30 pekerjaan='data scientist' skills=['Python', 'SQL', 'Machine Learning']

# Akses sebagai object
print(result.nama)      # Andi
print(result.skills)    # ['Python', 'SQL', 'Machine Learning']
```

### Format Instructions

```python
print(parser.get_format_instructions())
```

Output:

```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

{
  "nama": {"type": "string", "description": "Nama lengkap orang"},
  "umur": {"type": "integer", "description": "Umur dalam tahun"},
  ...
}
```

## Nested Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    """Alamat lengkap."""
    kota: str
    provinsi: str
    kode_pos: Optional[str] = None

class Company(BaseModel):
    """Informasi perusahaan."""
    nama: str
    industri: str
    alamat: Address
    jumlah_karyawan: int

parser = PydanticOutputParser(pydantic_object=Company)

# Usage sama seperti sebelumnya
```

## OutputFixingParser - Auto-Fix Errors

Jika parsing gagal, bisa auto-fix dengan bantuan LLM.

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# Base parser
base_parser = PydanticOutputParser(pydantic_object=Person)

# Wrap dengan OutputFixingParser
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# Jika LLM output tidak valid, akan otomatis di-fix
result = fixing_parser.parse(bad_output)
```

## RetryWithErrorOutputParser

Retry parsing dengan error message.

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# Akan retry dengan mengirim error message ke LLM
result = retry_parser.parse_with_prompt(
    completion=bad_output,
    prompt_value=original_prompt
)
```

## DatetimeOutputParser

```python
from langchain.output_parsers import DatetimeOutputParser

parser = DatetimeOutputParser()

template = ChatPromptTemplate.from_template(
    "When was {event}? {format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = template | ChatOpenAI(model="gpt-4o-mini") | parser

result = chain.invoke({"event": "Indonesia merdeka"})
print(result)  # datetime object: 1945-08-17 00:00:00
```

## EnumOutputParser

```python
from langchain.output_parsers import EnumOutputParser
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

parser = EnumOutputParser(enum=Sentiment)

template = ChatPromptTemplate.from_template("""
Analyze the sentiment of this text. 
{format_instructions}

Text: {text}
""").partial(format_instructions=parser.get_format_instructions())

chain = template | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser

result = chain.invoke({"text": "I love this product!"})
print(result)  # Sentiment.POSITIVE
```

## Error Handling

```python
from langchain_core.exceptions import OutputParserException

try:
    result = parser.parse(invalid_output)
except OutputParserException as e:
    print(f"Parsing failed: {e}")
    # Handle error: retry, fallback, or raise
```

## Practical Example: Product Review Analyzer

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ReviewAnalysis(BaseModel):
    """Analisis review produk."""
    sentiment: Sentiment = Field(description="Sentimen keseluruhan review")
    score: int = Field(description="Skor 1-5", ge=1, le=5)
    pros: list[str] = Field(description="Hal positif yang disebutkan")
    cons: list[str] = Field(description="Hal negatif yang disebutkan")
    summary: str = Field(description="Ringkasan 1 kalimat")

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

template = ChatPromptTemplate.from_template("""
Analyze this product review and extract structured information.

{format_instructions}

Review:
{review}
""")

chain = (
    template.partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | parser
)

# Test
review = """
Laptop ini bagus banget! Build quality-nya premium, keyboard-nya nyaman buat ngetik seharian.
Battery life juga awet, bisa 10 jam non-stop. Tapi agak berat sih untuk dibawa-bawa,
dan harganya memang mahal. Overall recommended untuk yang butuh laptop kerja profesional.
"""

result = chain.invoke({"review": review})

print(f"Sentiment: {result.sentiment.value}")
print(f"Score: {result.score}/5")
print(f"Pros: {result.pros}")
print(f"Cons: {result.cons}")
print(f"Summary: {result.summary}")
```

Output:

```
Sentiment: positive
Score: 4/5
Pros: ['Build quality premium', 'Keyboard nyaman', 'Battery life awet (10 jam)']
Cons: ['Berat untuk dibawa', 'Harga mahal']
Summary: Laptop berkualitas tinggi dengan build yang premium dan battery awet, cocok untuk profesional meski agak berat dan mahal.
```

## Pembandingan Parsers

| Parser | Output Type | Validation | Use Case |
|--------|------------|------------|----------|
| `StrOutputParser` | `str` | ❌ | Simple text |
| `JsonOutputParser` | `dict` | ❌ | Flexible JSON |
| `CommaSeparatedListOutputParser` | `list[str]` | ❌ | Simple lists |
| `PydanticOutputParser` | Pydantic Model | ✅ Full | Structured data |
| `EnumOutputParser` | Enum | ✅ | Classification |
| `DatetimeOutputParser` | datetime | ✅ | Dates |

## Ringkasan

1. **StrOutputParser** - extract string dari response
2. **JsonOutputParser** - parse JSON ke dict
3. **CommaSeparatedListOutputParser** - parse comma-separated ke list
4. **PydanticOutputParser** - dengan type validation dan schema
5. **OutputFixingParser** - auto-fix invalid output
6. Selalu **handle parsing errors** dengan try/except

---

**Selanjutnya:** [Menggabungkan Komponen](/docs/fondasi/menggabungkan-komponen) - Preview bagaimana semua komponen bekerja bersama.
