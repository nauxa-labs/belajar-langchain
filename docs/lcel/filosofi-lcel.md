---
sidebar_position: 1
title: Filosofi LCEL
description: Memahami paradigma deklaratif dan keuntungan LangChain Expression Language
---

# Filosofi LCEL

LCEL (LangChain Expression Language) adalah cara modern untuk membangun chains di LangChain. Ini adalah **pilar utama** framework yang membuat composing komponen menjadi mudah dan powerful.

## Apa itu LCEL?

LCEL adalah syntax deklaratif untuk menggabungkan komponen LangChain menggunakan **pipe operator** (`|`).

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LCEL Chain
chain = ChatPromptTemplate.from_template("Jelaskan {topik}") | ChatOpenAI() | StrOutputParser()

# Invoke
result = chain.invoke({"topik": "quantum computing"})
```

## Declarative vs Imperative

### Imperative (Gaya Lama)

Kamu harus menentukan **bagaimana** setiap langkah dieksekusi.

```python
# ❌ Imperative - banyak boilerplate
def translate(text: str, target: str) -> str:
    # Step 1: Format prompt
    prompt = f"Translate '{text}' to {target}"
    
    # Step 2: Create messages
    messages = [{"role": "user", "content": prompt}]
    
    # Step 3: Call API
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    # Step 4: Extract content
    content = response.choices[0].message.content
    
    return content
```

### Declarative (LCEL)

Kamu hanya menentukan **apa** yang ingin dilakukan.

```python
# ✅ Declarative - clean dan readable
chain = (
    ChatPromptTemplate.from_template("Translate '{text}' to {target}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

result = chain.invoke({"text": "Hello", "target": "Indonesian"})
```

## Kenapa Pipe Operator `|`?

Pipe operator diinspirasi dari Unix pipes dan functional programming:

```bash
# Unix pipe
cat file.txt | grep "error" | wc -l
```

```python
# LCEL pipe
prompt | llm | parser
```

Ini membuat data flow sangat jelas:
- Input masuk dari kiri
- Mengalir melalui setiap komponen
- Output keluar dari kanan

## Keuntungan LCEL

### 1. 🚀 Streaming Otomatis

Tanpa kode tambahan, kamu bisa stream output:

```python
chain = prompt | llm | parser

# Streaming gratis!
for chunk in chain.stream({"topik": "AI"}):
    print(chunk, end="")
```

### 2. ⚡ Async Out-of-the-Box

Semua chain otomatis support async:

```python
# Sync
result = chain.invoke(input)

# Async - sama syntax!
result = await chain.ainvoke(input)
```

### 3. 📦 Batching Built-in

Proses multiple inputs sekaligus:

```python
results = chain.batch([
    {"topik": "Python"},
    {"topik": "JavaScript"},
    {"topik": "Go"}
])
```

### 4. 🔄 Parallelization

Jalankan chains secara parallel:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    translation=translate_chain,
    keywords=keyword_chain
)

# Semua dijalankan bersamaan!
result = parallel.invoke({"text": "..."})
```

### 5. 🔁 Retry & Fallbacks

Error handling deklaratif:

```python
chain_with_fallback = main_chain.with_fallbacks([backup_chain])
chain_with_retry = main_chain.with_retry(stop_after_attempt=3)
```

### 6. 📊 Observability

Tracing otomatis dengan LangSmith:

```python
# Set environment variable
# LANGCHAIN_TRACING_V2=true

# Semua chain calls akan di-trace otomatis!
result = chain.invoke(input)
```

## Runnable Protocol

Semua komponen LCEL mengimplementasikan **Runnable Protocol**:

```python
class Runnable:
    def invoke(self, input) -> output
    def stream(self, input) -> Iterator[output]
    def batch(self, inputs) -> List[output]
    
    async def ainvoke(self, input) -> output
    async def astream(self, input) -> AsyncIterator[output]
    async def abatch(self, inputs) -> List[output]
```

Ini berarti **setiap komponen bisa diperlakukan sama** - baik itu prompt, LLM, parser, atau chain kompleks.

## Komponen yang Bisa Di-pipe

| Komponen | Input | Output |
|----------|-------|--------|
| `ChatPromptTemplate` | Dict | Messages |
| `ChatOpenAI` | Messages/String | AIMessage |
| `StrOutputParser` | AIMessage | String |
| `JsonOutputParser` | AIMessage | Dict |
| `RunnableLambda` | Any | Any |
| `RunnableParallel` | Dict | Dict |
| `RunnablePassthrough` | Any | Any (unchanged) |

## Contoh: Multi-step Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# Step 1: Generate
generate = ChatPromptTemplate.from_template(
    "Write a short story about {topic}"
) | ChatOpenAI() | StrOutputParser()

# Step 2: Improve
improve = ChatPromptTemplate.from_template(
    "Improve this story by adding more details:\n\n{story}"
) | ChatOpenAI() | StrOutputParser()

# Step 3: Translate
translate = ChatPromptTemplate.from_template(
    "Translate to Indonesian:\n\n{story}"
) | ChatOpenAI() | StrOutputParser()

# Combine with lambdas
full_chain = (
    generate
    | RunnableLambda(lambda x: {"story": x})
    | improve
    | RunnableLambda(lambda x: {"story": x})
    | translate
)

result = full_chain.invoke({"topic": "a robot learning to paint"})
print(result)
```

## Input & Output Schema

LCEL chains tahu input/output schema mereka:

```python
chain = prompt | llm | parser

# Input schema
print(chain.input_schema.schema())
# {'properties': {'topik': {'title': 'Topik', 'type': 'string'}}, 'required': ['topik']}

# Output schema
print(chain.output_schema.schema())
# {'title': 'StrOutputParserOutput', 'type': 'string'}
```

Ini membantu untuk:
- Validasi input
- Auto-generate documentation
- Type checking

## Kapan Tidak Pakai LCEL?

LCEL powerful tapi bukan untuk semua situasi:

| Situasi | Rekomendasi |
|---------|-------------|
| Simple API call | Langsung pakai SDK |
| Complex branching logic | Pertimbangkan LangGraph |
| Stateful workflows | Gunakan LangGraph |
| One-off scripts | LCEL atau langsung API |

## Visualizing Chains

Di Jupyter, kamu bisa visualisasi chain:

```python
# ASCII visualization
chain.get_graph().print_ascii()

# Mermaid diagram
print(chain.get_graph().draw_mermaid())
```

Output:

```
     +-----------+
     | Input     |
     +-----------+
           |
           v
+-------------------+
| ChatPromptTemplate|
+-------------------+
           |
           v
     +-----------+
     | ChatOpenAI|
     +-----------+
           |
           v
  +---------------+
  | StrOutputParser|
  +---------------+
           |
           v
     +-----------+
     | Output    |
     +-----------+
```

## Ringkasan

1. **LCEL** = LangChain Expression Language
2. **Pipe operator `|`** menghubungkan komponen
3. **Declarative** - fokus pada "apa", bukan "bagaimana"
4. **Built-in features**: streaming, async, batching, retry
5. Semua komponen mengimplementasikan **Runnable Protocol**
6. Chain composition membuat kode **readable dan maintainable**

---

**Selanjutnya:** [Runnable Interface](/docs/lcel/runnable-interface) - Deep dive ke methods yang tersedia di setiap Runnable.
