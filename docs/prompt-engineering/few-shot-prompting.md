---
sidebar_position: 2
title: Few-Shot Prompting
description: Menggunakan contoh untuk meningkatkan akurasi dan konsistensi output
---

# Few-Shot Prompting

Few-shot prompting adalah teknik memberikan beberapa contoh dalam prompt untuk "mengajarkan" LLM pola yang kita inginkan.

## Konsep Dasar

```
Zero-shot: Tidak ada contoh
One-shot: 1 contoh
Few-shot: 2-5 contoh (optimal)
Many-shot: 6+ contoh (jarang diperlukan)
```

## Mengapa Few-shot Efektif?

LLM belajar pattern recognition dari contoh:

```python
# Zero-shot - LLM harus guess format
prompt = "Classify the sentiment"
# Output bisa: "positive", "Positive", "POSITIVE", "pos", "😊"

# Few-shot - LLM follows pattern
prompt = """
Text: "I love this!" → Sentiment: positive
Text: "Terrible product" → Sentiment: negative
Text: "It's okay" → Sentiment: neutral
Text: "Best purchase ever!" → Sentiment:"""
# Output: positive (mengikuti pattern)
```

## LangChain FewShotPromptTemplate

### Basic Usage

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

# Template for each example
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Few-shot template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of the word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

# Format
prompt = few_shot_prompt.format(word="big")
print(prompt)
```

Output:

```
Give the antonym of the word.

Input: happy
Output: sad

Input: tall
Output: short

Input: fast
Output: slow

Input: big
Output:
```

## FewShotChatMessagePromptTemplate

Untuk Chat Models dengan proper message format.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# Examples as message pairs
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3*4", "output": "12"},
    {"input": "10/2", "output": "5"},
]

# Example template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Few-shot chat template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# Full prompt with system message
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math tutor."),
    few_shot_prompt,
    ("human", "{input}")
])

# Use
chain = final_prompt | llm | parser
result = chain.invoke({"input": "5+7"})
```

## Dynamic Example Selection

Tidak semua contoh relevan untuk setiap input. Gunakan **Example Selectors** untuk memilih contoh yang tepat.

### SemanticSimilarityExampleSelector

Pilih contoh berdasarkan semantic similarity.

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# All examples
examples = [
    {"input": "How do I add numbers?", "output": "Use the + operator: 2 + 3 = 5"},
    {"input": "How to create a list?", "output": "Use brackets: my_list = [1, 2, 3]"},
    {"input": "How to define a function?", "output": "Use def: def my_func(): pass"},
    {"input": "How to loop through items?", "output": "Use for: for item in list: print(item)"},
    {"input": "How to handle errors?", "output": "Use try/except: try: x=1/0 except: pass"},
]

# Create selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2  # Select 2 most similar examples
)

# Use in prompt
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Q: {input}\nA: {output}"
    ),
    prefix="Answer Python programming questions.",
    suffix="Q: {question}\nA:",
    input_variables=["question"]
)

# For "How to make a loop?" - akan pilih contoh tentang loop
prompt = dynamic_prompt.format(question="How to iterate over a dictionary?")
```

### MaxMarginalRelevanceExampleSelector

Pilih contoh yang relevan tapi juga diverse.

```python
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=3,
    fetch_k=10  # Fetch 10, then select 3 diverse ones
)
```

### LengthBasedExampleSelector

Pilih contoh berdasarkan budget token.

```python
from langchain_core.example_selectors import LengthBasedExampleSelector

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=1000  # Max characters
)
```

## Best Practices

### 1. Contoh yang Diverse

```python
# ❌ Bad - similar examples
examples = [
    {"text": "I love it!", "sentiment": "positive"},
    {"text": "I really love it!", "sentiment": "positive"},
    {"text": "I absolutely love it!", "sentiment": "positive"},
]

# ✅ Good - diverse examples
examples = [
    {"text": "I love it!", "sentiment": "positive"},
    {"text": "Terrible product.", "sentiment": "negative"},
    {"text": "It's okay I guess.", "sentiment": "neutral"},
]
```

### 2. Contoh yang Representative

```python
# Cover edge cases
examples = [
    # Normal case
    {"text": "Great product!", "sentiment": "positive"},
    # Mixed sentiment
    {"text": "Good but expensive", "sentiment": "mixed"},
    # Sarcasm
    {"text": "Yeah, 'great' service", "sentiment": "negative"},
    # Neutral
    {"text": "It arrived on Tuesday", "sentiment": "neutral"},
]
```

### 3. Format yang Konsisten

```python
# ✅ Consistent format
examples = [
    {"input": "hello", "output": "hola"},
    {"input": "goodbye", "output": "adiós"},
    {"input": "thank you", "output": "gracias"},
]

# ❌ Inconsistent format
examples = [
    {"input": "hello", "output": "hola"},
    {"word": "goodbye", "translation": "adiós"},  # Different keys!
    {"input": "thank you", "output": "Gracias."},  # Capitalized + period
]
```

### 4. Jumlah Optimal

```python
# Rule of thumb:
# - Simple tasks: 2-3 examples
# - Medium complexity: 3-5 examples
# - Complex/structured output: 5+ examples

# Jangan terlalu banyak - wastes tokens
# Jangan terlalu sedikit - tidak cukup pattern
```

## Practical Example: Code Translator

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Examples of Python to JavaScript translation
examples = [
    {
        "python": "def greet(name):\n    return f'Hello, {name}!'",
        "javascript": "function greet(name) {\n    return `Hello, ${name}!`;\n}"
    },
    {
        "python": "numbers = [1, 2, 3]\nfor n in numbers:\n    print(n)",
        "javascript": "const numbers = [1, 2, 3];\nfor (const n of numbers) {\n    console.log(n);\n}"
    },
    {
        "python": "class Person:\n    def __init__(self, name):\n        self.name = name",
        "javascript": "class Person {\n    constructor(name) {\n        this.name = name;\n    }\n}"
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Python:\n```python\n{python}\n```"),
    ("ai", "JavaScript:\n```javascript\n{javascript}\n```")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

full_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at converting Python code to JavaScript. Maintain the same logic and structure."),
    few_shot,
    ("human", "Python:\n```python\n{code}\n```")
])

chain = full_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

# Test
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

result = chain.invoke({"code": python_code})
print(result)
```

## Ringkasan

1. **Few-shot** = berikan contoh dalam prompt
2. **2-5 examples** biasanya optimal
3. **FewShotPromptTemplate** untuk string prompts
4. **FewShotChatMessagePromptTemplate** untuk chat models
5. **Example Selectors** untuk dynamic selection
6. Pastikan contoh **diverse** dan **consistent**

---

**Selanjutnya:** [Advanced Techniques](/docs/prompt-engineering/advanced-techniques) - Chain-of-Thought dan teknik lanjutan.
