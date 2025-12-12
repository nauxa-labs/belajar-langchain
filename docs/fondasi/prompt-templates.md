---
sidebar_position: 3
title: Prompt Templates
description: Membuat prompt yang reusable dan maintainable dengan LangChain
---

# Prompt Templates

Prompt Templates adalah salah satu building block paling penting di LangChain. Mereka memungkinkan kita membuat prompt yang **reusable**, **testable**, dan **maintainable**.

## Mengapa Tidak Pakai F-String?

```python
# ❌ Plain f-string - tidak ideal
def translate(text: str, target_lang: str) -> str:
    prompt = f"Translate '{text}' to {target_lang}"
    return llm.invoke(prompt)
```

**Masalah dengan f-string:**
1. Tidak bisa di-serialize/share
2. Sulit di-test secara terpisah
3. Tidak ada validation untuk variables
4. Tidak bisa digunakan dengan LCEL chains

## PromptTemplate - Untuk String Output

`PromptTemplate` menghasilkan string prompt.

### Basic Usage

```python
from langchain_core.prompts import PromptTemplate

# Buat template
template = PromptTemplate.from_template(
    "Terjemahkan teks berikut ke {bahasa}: {teks}"
)

# Format dengan variables
prompt = template.format(bahasa="Inggris", teks="Selamat pagi")
print(prompt)
# Output: "Terjemahkan teks berikut ke Inggris: Selamat pagi"

# Invoke dengan values
result = template.invoke({"bahasa": "Inggris", "teks": "Selamat pagi"})
print(result.text)
# Output sama seperti di atas
```

### Dengan Explicit Input Variables

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="Jelaskan {topik} untuk {audiens} dalam {jumlah} kalimat.",
    input_variables=["topik", "audiens", "jumlah"]
)

prompt = template.format(
    topik="machine learning",
    audiens="anak SD",
    jumlah="3"
)
print(prompt)
```

Output:

```
Jelaskan machine learning untuk anak SD dalam 3 kalimat.
```

## ChatPromptTemplate - Untuk Chat Models

`ChatPromptTemplate` menghasilkan list of messages, cocok untuk Chat Models modern.

### Basic Chat Template

```python
from langchain_core.prompts import ChatPromptTemplate

# Buat template dengan system dan human messages
template = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah asisten yang ahli dalam {bidang}."),
    ("human", "{pertanyaan}")
])

# Format
messages = template.format_messages(
    bidang="programming",
    pertanyaan="Apa itu recursion?"
)

# Result: List of message objects
for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

Output:

```
system: Kamu adalah asisten yang ahli dalam programming.
human: Apa itu recursion?
```

### Message Types dalam Template

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    # System message - instruksi untuk AI
    ("system", "Kamu adalah {persona}. Jawab dengan gaya {gaya}."),
    
    # Placeholder untuk conversation history
    MessagesPlaceholder(variable_name="history"),
    
    # Human message - input user
    ("human", "{input}"),
])

# Usage dengan history
from langchain_core.messages import HumanMessage, AIMessage

messages = template.format_messages(
    persona="guru matematika",
    gaya="santai",
    history=[
        HumanMessage(content="Halo!"),
        AIMessage(content="Halo juga! Ada yang bisa dibantu?")
    ],
    input="Apa itu bilangan prima?"
)
```

### Shorthand Syntax

```python
from langchain_core.prompts import ChatPromptTemplate

# Tuple syntax (type, content)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
    ("ai", "Let me think about that..."),  # Optional AI response template
    ("human", "Can you elaborate?")
])
```

## Partial Prompts

Kadang kita ingin "pre-fill" beberapa variable.

### Partial dengan Values

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Jelaskan {topik} untuk {audiens}"
)

# Pre-fill 'audiens'
template_for_beginners = template.partial(audiens="pemula")

# Sekarang hanya butuh 'topik'
prompt = template_for_beginners.format(topik="API REST")
print(prompt)
# Output: "Jelaskan API REST untuk pemula"
```

### Partial dengan Functions

```python
from datetime import datetime

def get_current_date() -> str:
    return datetime.now().strftime("%d %B %Y")

template = PromptTemplate(
    template="Hari ini tanggal {tanggal}. Jelaskan event yang terjadi pada {event_date}.",
    partial_variables={"tanggal": get_current_date}
)

# 'tanggal' akan di-fill otomatis
prompt = template.format(event_date="17 Agustus 1945")
print(prompt)
# Output: "Hari ini tanggal 12 Desember 2024. Jelaskan event yang terjadi pada 17 Agustus 1945."
```

## Template Composition

Menggabungkan beberapa template.

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Define individual templates
system_template = "Kamu adalah {role}."
instruction_template = "Tugas: {task}\n\nInput: {input}"

# Combine into ChatPromptTemplate
full_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", instruction_template)
])

messages = full_template.format_messages(
    role="penerjemah profesional",
    task="Terjemahkan teks ke Bahasa Inggris",
    input="Selamat datang di Indonesia!"
)
```

## Template dengan LCEL

Templates sangat powerful ketika digabung dengan LCEL (LangChain Expression Language).

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define chain
template = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah penerjemah dari Bahasa Indonesia ke {bahasa_target}."),
    ("human", "Terjemahkan: {teks}")
])

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Chain dengan pipe operator
chain = template | llm | parser

# Invoke chain
result = chain.invoke({
    "bahasa_target": "Bahasa Jepang",
    "teks": "Selamat pagi, apa kabar?"
})

print(result)
# Output: おはようございます、お元気ですか？
```

## Validasi Variables

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You speak {language}"),
    ("human", "{message}")
])

# Check required variables
print(template.input_variables)
# Output: ['language', 'message']

# Error jika variable missing
try:
    template.format_messages(language="English")  # Missing 'message'
except KeyError as e:
    print(f"Missing variable: {e}")
```

## Best Practices

### 1. Gunakan System Message dengan Baik

```python
# ✅ Good - specific dan actionable
system_template = """
Kamu adalah code reviewer senior dengan keahlian di:
- Python best practices
- Clean code principles
- Security vulnerabilities

Saat mereview code:
1. Identifikasi potential bugs
2. Suggest improvements
3. Rate security (1-10)
4. Berikan contoh perbaikan jika perlu

Format response sebagai markdown.
"""

# ❌ Bad - terlalu vague
system_template = "You are a code reviewer"
```

### 2. Dokumentasikan Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# Template dengan dokumentasi
code_review_template = ChatPromptTemplate.from_messages([
    ("system", """
    Code Review Assistant
    ---------------------
    Purpose: Review Python code for quality and security
    
    Output Format:
    - Bugs: List of potential issues
    - Improvements: Suggestions for better code
    - Security: Rating 1-10 with explanation
    """),
    ("human", "Review this code:\n```python\n{code}\n```")
])
```

### 3. Template Variables yang Jelas

```python
# ✅ Good - nama variable deskriptif
template = ChatPromptTemplate.from_template(
    "Write a {content_type} about {topic} for {target_audience}"
)

# ❌ Bad - nama variable tidak jelas
template = ChatPromptTemplate.from_template(
    "Write a {x} about {y} for {z}"
)
```

### 4. Gunakan Default Values

```python
from langchain_core.prompts import ChatPromptTemplate

# Template dengan default
template = ChatPromptTemplate.from_messages([
    ("system", "Output format: {format}"),
    ("human", "{question}")
]).partial(format="plain text")

# Bisa override atau pakai default
result1 = template.format_messages(question="What is 2+2?")
result2 = template.format_messages(question="What is 2+2?", format="JSON")
```

## Contoh Praktis: Multi-purpose Template

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Template yang flexible
assistant_template = ChatPromptTemplate.from_messages([
    ("system", """
    Kamu adalah {role}.
    
    Gaya komunikasi: {style}
    Bahasa output: {language}
    
    Panduan tambahan:
    {guidelines}
    """),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{input}")
])

# Factory function untuk membuat specialized chains
def create_assistant(role: str, style: str, guidelines: str = "Tidak ada"):
    return (
        assistant_template.partial(
            role=role,
            style=style,
            language="Bahasa Indonesia",
            guidelines=guidelines
        )
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

# Create specialized assistants
code_helper = create_assistant(
    role="senior Python developer",
    style="teknis tapi mudah dipahami",
    guidelines="Selalu sertakan code example"
)

writing_helper = create_assistant(
    role="penulis konten profesional",
    style="engaging dan informatif",
    guidelines="Gunakan heading dan bullet points"
)

# Usage
code_result = code_helper.invoke({"input": "Jelaskan list comprehension di Python"})
writing_result = writing_helper.invoke({"input": "Tulis intro artikel tentang AI"})
```

## Ringkasan

1. **PromptTemplate** - untuk string output (legacy/simple use cases)
2. **ChatPromptTemplate** - untuk chat models (modern standard)
3. **MessagesPlaceholder** - untuk menyisipkan conversation history
4. **Partial prompts** - pre-fill variables yang konstan
5. **LCEL integration** - template | llm | parser pattern
6. Gunakan **nama variable yang jelas** dan **dokumentasi**

---

**Selanjutnya:** [Output Parsers](/docs/fondasi/output-parsers) - Cara memproses output LLM menjadi format yang terstruktur.
