---
sidebar_position: 2
title: Memanggil Model Pertama
description: Berbagai cara memanggil Chat Models di LangChain
---

# Memanggil Model Pertama

Sekarang kita akan mempelajari berbagai cara memanggil Chat Models di LangChain, termasuk synchronous, asynchronous, dan streaming calls.

## Setup

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Inisialisasi model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
```

## Metode Pemanggilan

### 1. `invoke()` - Synchronous

Cara paling dasar dan paling sering digunakan.

```python
# Dengan string (shorthand)
response = llm.invoke("Apa itu Python?")
print(response.content)

# Dengan messages (full control)
messages = [
    SystemMessage(content="Jawab dengan singkat dalam Bahasa Indonesia."),
    HumanMessage(content="Apa itu Python?")
]
response = llm.invoke(messages)
print(response.content)
```

Output:

```
Python adalah bahasa pemrograman tingkat tinggi yang mudah dipelajari, 
dengan sintaks yang bersih dan banyak digunakan untuk web development, 
data science, AI, dan automation.
```

### 2. `stream()` - Streaming Output

Menampilkan output token per token (seperti ChatGPT).

```python
# Streaming - output muncul bertahap
for chunk in llm.stream("Ceritakan tentang Indonesia dalam 3 kalimat"):
    print(chunk.content, end="", flush=True)
print()  # New line di akhir
```

Output (muncul bertahap):

```
Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau. 
Negara ini memiliki keanekaragaman budaya, bahasa, dan tradisi yang luar biasa. 
Indonesia juga dikenal dengan keindahan alamnya, dari pantai Bali hingga hutan 
Kalimantan.
```

:::tip Kapan Pakai Streaming?
- **Real-time chat applications** - user experience lebih baik
- **Long responses** - user tidak perlu menunggu seluruh response
- **Progress indication** - menunjukkan AI sedang "berpikir"
:::

### 3. `batch()` - Multiple Inputs

Memproses beberapa input sekaligus.

```python
# Batch processing
questions = [
    "Apa ibukota Indonesia?",
    "Apa ibukota Malaysia?",
    "Apa ibukota Thailand?"
]

# Convert ke messages
messages_list = [[HumanMessage(content=q)] for q in questions]

# Batch invoke
responses = llm.batch(messages_list)

for q, r in zip(questions, responses):
    print(f"Q: {q}")
    print(f"A: {r.content}\n")
```

Output:

```
Q: Apa ibukota Indonesia?
A: Ibukota Indonesia adalah Jakarta.

Q: Apa ibukota Malaysia?
A: Ibukota Malaysia adalah Kuala Lumpur.

Q: Apa ibukota Thailand?
A: Ibukota Thailand adalah Bangkok.
```

### 4. Async Methods

Untuk aplikasi async seperti FastAPI.

```python
import asyncio

async def async_example():
    # Async invoke
    response = await llm.ainvoke("Halo, apa kabar?")
    print(response.content)
    
    # Async stream
    async for chunk in llm.astream("Ceritakan joke singkat"):
        print(chunk.content, end="")
    print()

# Run
asyncio.run(async_example())
```

### 5. `abatch()` - Async Batch

```python
async def async_batch_example():
    messages_list = [
        [HumanMessage(content="Ibu kota Jepang?")],
        [HumanMessage(content="Ibu kota Korea Selatan?")],
        [HumanMessage(content="Ibu kota China?")]
    ]
    
    responses = await llm.abatch(messages_list)
    
    for r in responses:
        print(r.content)

asyncio.run(async_batch_example())
```

## Berbagai LLM Providers

### OpenAI

```python
from langchain_openai import ChatOpenAI

# GPT-4o Mini - cepat dan murah
llm = ChatOpenAI(model="gpt-4o-mini")

# GPT-4o - lebih capable
llm = ChatOpenAI(model="gpt-4o")

# GPT-4 Turbo
llm = ChatOpenAI(model="gpt-4-turbo")

# Dengan custom parameters
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=500,
    timeout=30,  # seconds
    max_retries=2
)
```

### Anthropic (Claude)

```python
from langchain_anthropic import ChatAnthropic

# Claude 3 Haiku - tercepat
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# Claude 3 Sonnet - balanced
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Claude 3 Opus - paling capable
llm = ChatAnthropic(model="claude-3-opus-20240229")

# Claude 3.5 Sonnet - terbaru
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

### Google (Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini Pro
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Gemini 1.5 Pro
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Gemini 1.5 Flash - cepat
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
```

### Ollama (Local/Open Source)

```python
from langchain_ollama import ChatOllama

# Llama 3
llm = ChatOllama(model="llama3")

# Mistral
llm = ChatOllama(model="mistral")

# Custom endpoint
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434"
)
```

## Response Object

Mari explore object response yang dikembalikan:

```python
response = llm.invoke("Halo!")

# Type
print(type(response))
# <class 'langchain_core.messages.ai.AIMessage'>

# Content
print(response.content)
# "Halo! Apa kabar? Ada yang bisa saya bantu hari ini?"

# Response metadata (varies by provider)
print(response.response_metadata)
# {
#     'token_usage': {'completion_tokens': 15, 'prompt_tokens': 10, 'total_tokens': 25},
#     'model_name': 'gpt-4o-mini',
#     'finish_reason': 'stop'
# }

# Usage info
print(response.usage_metadata)
# {'input_tokens': 10, 'output_tokens': 15, 'total_tokens': 25}
```

## Error Handling

```python
from langchain_core.exceptions import OutputParserException
import openai

def safe_invoke(llm, message: str) -> str:
    """Safely invoke LLM with error handling."""
    try:
        response = llm.invoke(message)
        return response.content
    except openai.RateLimitError:
        print("Rate limit hit. Please wait...")
        return None
    except openai.APIError as e:
        print(f"API Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
result = safe_invoke(llm, "Hello!")
if result:
    print(result)
```

### Retry dengan Tenacity

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def invoke_with_retry(llm, message: str) -> str:
    response = llm.invoke(message)
    return response.content

# Will automatically retry up to 3 times with exponential backoff
result = invoke_with_retry(llm, "Hello!")
```

## Timeout Configuration

```python
# Set timeout at initialization
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30.0,      # 30 seconds
    max_retries=2      # Retry 2 times on failure
)

# Or per-request (using httpx)
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke(
    "Complex question...",
    config={"timeout": 60}  # 60 seconds for this request
)
```

## Practical Example: Simple Chatbot

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def simple_chatbot():
    """Simple command-line chatbot."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Initialize with system message
    messages = [
        SystemMessage(content="""
        Kamu adalah asisten AI yang ramah bernama Langchain Bot.
        Jawab dalam Bahasa Indonesia dengan gaya casual tapi informatif.
        """)
    ]
    
    print("🤖 Langchain Bot siap! (ketik 'quit' untuk keluar)\n")
    
    while True:
        user_input = input("Kamu: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("👋 Sampai jumpa!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Get response with streaming
        print("Bot: ", end="")
        full_response = ""
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
        print("\n")
        
        # Add AI response to history
        messages.append(AIMessage(content=full_response))

if __name__ == "__main__":
    simple_chatbot()
```

## Perbandingan Methods

| Method | Sync/Async | Use Case |
|--------|-----------|----------|
| `invoke()` | Sync | Single request, simple scripts |
| `stream()` | Sync | Real-time output, chat UI |
| `batch()` | Sync | Multiple requests, bulk processing |
| `ainvoke()` | Async | Web servers (FastAPI, etc) |
| `astream()` | Async | Async real-time output |
| `abatch()` | Async | Async bulk processing |

## Ringkasan

1. **`invoke()`** - cara paling dasar untuk single call
2. **`stream()`** - untuk real-time token streaming
3. **`batch()`** - untuk multiple inputs sekaligus
4. **Async variants** (`ainvoke`, `astream`, `abatch`) untuk async apps
5. Berbagai **providers** tersedia: OpenAI, Anthropic, Google, Ollama
6. Jangan lupa **error handling** dan **timeouts**

---

**Selanjutnya:** [Prompt Templates](/docs/fondasi/prompt-templates) - Cara membuat prompt yang reusable dan maintainable.
