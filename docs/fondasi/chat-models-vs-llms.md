---
sidebar_position: 1
title: Chat Models vs LLMs
description: Memahami perbedaan completion model dan chat model di LangChain
---

# Chat Models vs LLMs

Di LangChain, ada dua jenis model utama: **LLMs** dan **Chat Models**. Memahami perbedaan ini penting karena akan mempengaruhi bagaimana kamu berinteraksi dengan model.

## Perbedaan Fundamental

### LLMs (Completion Models)

Model yang menerima **string** dan menghasilkan **string**.

```python
# Input: "The capital of Indonesia is"
# Output: " Jakarta, a bustling metropolis..."
```

**Karakteristik:**
- Input: Plain text
- Output: Plain text (completion)
- Contoh: GPT-3, text-davinci-003 (legacy)

### Chat Models

Model yang menerima **list of messages** dan menghasilkan **message**.

```python
# Input: [
#   SystemMessage("You are a helpful assistant"),
#   HumanMessage("What is the capital of Indonesia?")
# ]
# Output: AIMessage("The capital of Indonesia is Jakarta.")
```

**Karakteristik:**
- Input: List of messages dengan roles
- Output: Message object
- Contoh: GPT-4, Claude 3, Gemini

:::tip Fokus ke Chat Models
Hampir semua model modern adalah Chat Models. Di kurikulum ini, kita akan fokus menggunakan Chat Models karena lebih versatile dan merupakan standar industri saat ini.
:::

## Message Types

Chat Models menggunakan sistem messages dengan roles berbeda:

### 1. SystemMessage

Instruksi untuk model tentang behavior dan persona.

```python
from langchain_core.messages import SystemMessage

system = SystemMessage(content="""
You are a helpful programming assistant.
You write clean, well-documented Python code.
Always explain your code briefly.
""")
```

### 2. HumanMessage

Pesan dari user (manusia).

```python
from langchain_core.messages import HumanMessage

human = HumanMessage(content="Write a function to calculate factorial")
```

### 3. AIMessage

Response dari AI assistant.

```python
from langchain_core.messages import AIMessage

ai = AIMessage(content="Here's a factorial function...")
```

### 4. ToolMessage (untuk Agents)

Response dari tool/function call.

```python
from langchain_core.messages import ToolMessage

tool = ToolMessage(
    content="Weather in Jakarta: 32°C, Sunny",
    tool_call_id="weather_123"
)
```

## Contoh Praktis

### Basic Chat

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Inisialisasi model
llm = ChatOpenAI(model="gpt-4o-mini")

# Buat messages
messages = [
    SystemMessage(content="Kamu adalah asisten yang ramah dan membantu. Jawab dalam Bahasa Indonesia."),
    HumanMessage(content="Apa itu LangChain?")
]

# Panggil model
response = llm.invoke(messages)

print(response.content)
```

Output:

```
LangChain adalah framework Python yang memudahkan pengembangan aplikasi 
berbasis Large Language Model (LLM). Framework ini menyediakan:

1. Abstraksi untuk berbagai LLM providers (OpenAI, Anthropic, dll)
2. Tools untuk membangun chains dan pipelines
3. Integrasi dengan vector databases untuk RAG
4. Agent framework untuk membuat AI yang bisa menggunakan tools

LangChain sangat populer untuk membangun chatbots, QA systems, dan 
aplikasi AI lainnya.
```

### Multi-turn Conversation

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini")

# Simulasi percakapan multi-turn
messages = [
    SystemMessage(content="Kamu adalah guru matematika yang sabar."),
    HumanMessage(content="Apa itu bilangan prima?"),
    AIMessage(content="Bilangan prima adalah bilangan yang hanya bisa dibagi 1 dan dirinya sendiri. Contoh: 2, 3, 5, 7, 11..."),
    HumanMessage(content="Apakah 9 bilangan prima?")
]

response = llm.invoke(messages)
print(response.content)
```

Output:

```
Tidak, 9 bukan bilangan prima. 

9 bisa dibagi oleh:
- 1
- 3 (karena 9 = 3 × 3)
- 9

Karena 9 bisa dibagi oleh 3 selain 1 dan dirinya sendiri, 
maka 9 termasuk bilangan komposit, bukan prima.
```

## Parameter Penting

### temperature

Mengontrol "kreativitas" output.

```python
# temperature=0: Deterministik, konsisten
llm_precise = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# temperature=0.7: Balanced (default)
llm_balanced = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# temperature=1.0: Lebih kreatif, varied
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
```

| Nilai | Use Case |
|-------|----------|
| 0 | Kode, fakta, klasifikasi |
| 0.3-0.7 | Chatbot general |
| 0.8-1.0 | Creative writing, brainstorming |

### max_tokens

Batas maksimum tokens di output.

```python
# Batasi output ke 100 tokens
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
```

:::info Apa itu Token?
Token adalah unit dasar yang diproses LLM. Secara kasar:
- 1 token ≈ 4 karakter dalam Bahasa Inggris
- 1 token ≈ 2-3 karakter dalam Bahasa Indonesia
- "Hello world" = 2 tokens
- "Selamat pagi" = 3-4 tokens
:::

### top_p (Nucleus Sampling)

Alternative ke temperature untuk mengontrol randomness.

```python
# top_p=0.1: Hanya consider top 10% tokens
llm = ChatOpenAI(model="gpt-4o-mini", top_p=0.1)

# top_p=0.9: Consider top 90% tokens (default)
llm = ChatOpenAI(model="gpt-4o-mini", top_p=0.9)
```

:::tip Praktis
Biasanya cukup tune `temperature` saja. Jangan ubah `temperature` dan `top_p` bersamaan.
:::

### Model Selection

```python
# OpenAI models
llm_fast = ChatOpenAI(model="gpt-4o-mini")      # Cepat, murah
llm_smart = ChatOpenAI(model="gpt-4o")          # Lebih pintar
llm_turbo = ChatOpenAI(model="gpt-4-turbo")     # Balance

# Anthropic models
from langchain_anthropic import ChatAnthropic
llm_haiku = ChatAnthropic(model="claude-3-haiku-20240307")   # Cepat
llm_sonnet = ChatAnthropic(model="claude-3-sonnet-20240229") # Balance
llm_opus = ChatAnthropic(model="claude-3-opus-20240229")     # Paling pintar
```

## Best Practices

### 1. System Message yang Jelas

```python
# ❌ Kurang spesifik
system = SystemMessage(content="You are helpful")

# ✅ Spesifik dan detail
system = SystemMessage(content="""
You are a senior Python developer with expertise in:
- Clean code principles
- Test-driven development
- API design

When writing code:
1. Include type hints
2. Add docstrings
3. Handle errors gracefully
4. Keep functions small and focused
""")
```

### 2. Gunakan temperature Sesuai Use Case

```python
# Untuk data extraction - butuh konsistensi
extractor = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Untuk creative writing
writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
```

### 3. Handle Response dengan Benar

```python
response = llm.invoke(messages)

# Response adalah AIMessage object
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>

# Akses content
print(response.content)  # String content

# Akses metadata
print(response.response_metadata)  # Token usage, model info, etc.
```

## Perbandingan Interface

| Aspect | LLM | Chat Model |
|--------|-----|------------|
| Input | String | List[Message] |
| Output | String | AIMessage |
| Context | Manual | Via message history |
| System prompt | Prepend to input | SystemMessage |
| Modern support | Limited | Full |

## Ringkasan

1. **Chat Models** adalah standar modern - fokus di sini
2. Messages memiliki **roles**: System, Human, AI, Tool
3. **System message** mengatur behavior model
4. **temperature** mengontrol kreativitas (0=deterministic, 1=creative)
5. **max_tokens** membatasi panjang output

---

**Selanjutnya:** [Memanggil Model Pertama](/docs/fondasi/memanggil-model) - Kita akan explore berbagai cara memanggil Chat Models.
