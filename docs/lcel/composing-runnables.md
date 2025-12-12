---
sidebar_position: 3
title: Composing Runnables
description: Menggabungkan runnables dengan RunnableSequence, RunnableParallel, dan RunnablePassthrough
---

# Composing Runnables

Kekuatan LCEL ada di kemampuan **composing** - menggabungkan komponen sederhana menjadi pipeline kompleks. Di bab ini kita akan belajar berbagai cara menggabungkan runnables.

## RunnableSequence (Pipe Operator)

Cara paling dasar - menghubungkan komponen secara sekuensial dengan `|`.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Ini adalah RunnableSequence
chain = (
    ChatPromptTemplate.from_template("Jelaskan {topik}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Data flow:
# {"topik": "AI"} → Prompt → Messages → LLM → AIMessage → Parser → String
```

### Explicit RunnableSequence

```python
from langchain_core.runnables import RunnableSequence

# Equivalent dengan pipe operator
chain = RunnableSequence(
    first=ChatPromptTemplate.from_template("Jelaskan {topik}"),
    middle=[ChatOpenAI(model="gpt-4o-mini")],
    last=StrOutputParser()
)
```

## RunnableParallel

Menjalankan beberapa chains **secara bersamaan** dan menggabungkan hasilnya.

```python
from langchain_core.runnables import RunnableParallel

# Define individual chains
summary_chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

translate_chain = (
    ChatPromptTemplate.from_template("Translate to Indonesian: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

keywords_chain = (
    ChatPromptTemplate.from_template("Extract 5 keywords from: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Run in parallel
parallel = RunnableParallel(
    summary=summary_chain,
    translation=translate_chain,
    keywords=keywords_chain
)

result = parallel.invoke({"text": "LangChain is a framework for building LLM applications..."})

print(result)
# {
#     "summary": "LangChain adalah framework...",
#     "translation": "LangChain adalah kerangka kerja...",
#     "keywords": "LangChain, framework, LLM, applications, building"
# }
```

### Shorthand dengan Dict

```python
# Dict syntax - otomatis jadi RunnableParallel
parallel = {
    "summary": summary_chain,
    "translation": translate_chain,
    "keywords": keywords_chain
}

# Bisa langsung di-pipe
full_chain = some_input | parallel | some_output
```

### Parallel dengan Input Transformation

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Transform input untuk different chains
parallel = RunnableParallel(
    original=RunnablePassthrough(),  # Keep original input
    uppercase=lambda x: x.upper(),
    length=lambda x: len(x)
)

result = parallel.invoke("hello world")
# {"original": "hello world", "uppercase": "HELLO WORLD", "length": 11}
```

## RunnablePassthrough

Meneruskan input tanpa modifikasi - berguna untuk menyisipkan data.

### Basic Passthrough

```python
from langchain_core.runnables import RunnablePassthrough

# Pass input unchanged
chain = RunnablePassthrough() | some_function

result = chain.invoke("test")  # "test" diteruskan ke some_function
```

### Passthrough dengan Assign

Menambahkan field baru tanpa menghapus yang lama:

```python
from langchain_core.runnables import RunnablePassthrough

# Add new field while keeping existing ones
chain = RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split())
)

result = chain.invoke({"text": "Hello world from LangChain"})
# {"text": "Hello world from LangChain", "word_count": 4}
```

### Common Pattern: RAG

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def get_context(query: str) -> str:
    # Simulate retrieval
    return "Retrieved context for: " + query

# RAG pattern dengan passthrough
rag_chain = (
    RunnableParallel(
        context=lambda x: get_context(x["question"]),
        question=lambda x: x["question"]
    )
    | ChatPromptTemplate.from_template("""
        Context: {context}
        Question: {question}
        Answer:
    """)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "What is LangChain?"})
```

## RunnableLambda

Membungkus Python function sebagai Runnable.

```python
from langchain_core.runnables import RunnableLambda

def process_text(text: str) -> str:
    return text.strip().upper()

# Wrap as Runnable
process_runnable = RunnableLambda(process_text)

# Atau gunakan decorator
@RunnableLambda
def process_text_v2(text: str) -> str:
    return text.strip().upper()

# Bisa di-pipe
chain = some_chain | process_runnable | another_chain
```

### Lambda dengan Multiple Arguments

```python
from langchain_core.runnables import RunnableLambda

def format_response(data: dict) -> str:
    return f"Title: {data['title']}\nContent: {data['content']}"

chain = (
    some_chain
    | RunnableLambda(format_response)
)
```

### Async Lambda

```python
async def async_process(text: str) -> str:
    await asyncio.sleep(1)  # Simulate async operation
    return text.upper()

# Otomatis support sync dan async
runnable = RunnableLambda(async_process)

# Sync (akan run dalam thread)
result = runnable.invoke("test")

# Async
result = await runnable.ainvoke("test")
```

## Combining Patterns

### Multi-step Pipeline

```python
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

# Step 1: Parallel processing
parallel_step = RunnableParallel(
    summary=ChatPromptTemplate.from_template("Summarize: {text}") | llm | parser,
    sentiment=ChatPromptTemplate.from_template("Sentiment of: {text}") | llm | parser
)

# Step 2: Combine results
def combine_results(data: dict) -> dict:
    return {
        "combined": f"Summary: {data['summary']}\nSentiment: {data['sentiment']}"
    }

# Step 3: Final processing
final_step = (
    ChatPromptTemplate.from_template("Based on this analysis: {combined}, give recommendation")
    | llm
    | parser
)

# Full pipeline
full_pipeline = (
    {"text": RunnablePassthrough()}  # Wrap input
    | parallel_step
    | RunnableLambda(combine_results)
    | final_step
)

result = full_pipeline.invoke("LangChain is amazing for building AI apps...")
```

### Chaining Parallels

```python
# First parallel: analyze
analyze = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain
)

# Second parallel: translate both results
translate = RunnableParallel(
    summary_id=ChatPromptTemplate.from_template("Translate to ID: {summary}") | llm | parser,
    keywords_id=ChatPromptTemplate.from_template("Translate to ID: {keywords}") | llm | parser
)

# Chain them
full_chain = analyze | translate
```

## Input/Output Transformations

### itemgetter untuk Dict Access

```python
from operator import itemgetter

chain = (
    {
        "context": itemgetter("context"),
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | parser
)

result = chain.invoke({
    "context": "LangChain is a framework...",
    "question": "What is LangChain?"
})
```

### Custom Transformations

```python
from langchain_core.runnables import RunnableLambda

# Transform input
preprocess = RunnableLambda(lambda x: {
    "text": x["raw_text"].strip(),
    "timestamp": datetime.now().isoformat()
})

# Transform output
postprocess = RunnableLambda(lambda x: {
    "result": x,
    "status": "success"
})

chain = preprocess | main_chain | postprocess
```

## Debugging Compositions

### Visualize Chain

```python
# Print ASCII graph
chain.get_graph().print_ascii()

# Get Mermaid diagram
mermaid = chain.get_graph().draw_mermaid()
print(mermaid)
```

### Inspect Input/Output

```python
from langchain_core.runnables import RunnableLambda

def debug_step(data):
    print(f"Debug: {type(data)} = {data}")
    return data

debug = RunnableLambda(debug_step)

# Insert debug anywhere in chain
chain = step1 | debug | step2 | debug | step3
```

## Performance Considerations

### Parallel vs Sequential

```python
# Sequential - slow
chain = step1 | step2 | step3  # Total: sum of all steps

# Parallel - fast
parallel = RunnableParallel(a=step1, b=step2, c=step3)  # Total: max of all steps
```

### When to Use Parallel

| Use Parallel | Use Sequential |
|--------------|----------------|
| Independent operations | Output depends on previous |
| Multiple API calls | Step-by-step processing |
| Aggregating multiple sources | Data transformation pipeline |

## Ringkasan

1. **RunnableSequence** (`|`) - eksekusi sekuensial
2. **RunnableParallel** - eksekusi paralel
3. **RunnablePassthrough** - teruskan input unchanged
4. **RunnablePassthrough.assign()** - tambah field baru
5. **RunnableLambda** - wrap Python function
6. **Dict syntax** - shorthand untuk RunnableParallel
7. Combine patterns untuk **pipelines kompleks**

---

**Selanjutnya:** [Branching & Routing](/docs/lcel/branching-routing) - Conditional execution dan dynamic routing.
