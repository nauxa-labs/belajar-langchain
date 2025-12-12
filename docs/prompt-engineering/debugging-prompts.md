---
sidebar_position: 5
title: Debugging Prompts
description: Teknik debugging, logging, dan iterasi prompts
---

# Debugging Prompts

Prompt engineering adalah proses iteratif. Bab ini membahas cara debugging dan improve prompts secara sistematis.

## Verbose Mode

Aktifkan verbose logging untuk melihat apa yang terjadi.

```python
from langchain.globals import set_verbose, set_debug

# Verbose - show main steps
set_verbose(True)

# Debug - show ALL details (very verbose)
set_debug(True)

# Now run your chain
result = chain.invoke(dict(input="test"))
```

Output:

```text
[chain/start] [1:chain:RunnableSequence] Entering Chain run with input:
[llm/start] [1:chain:RunnableSequence > 2:llm:ChatOpenAI] Entering LLM run
[llm/end] Exiting LLM run with output
[chain/end] [1:chain:RunnableSequence] Exiting Chain run with output
```

## Callbacks untuk Custom Logging

```python
from langchain_core.callbacks import BaseCallbackHandler

class PromptDebugHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("=" * 50)
        print("📤 PROMPT SENT TO LLM:")
        print("=" * 50)
        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1} ---")
            print(prompt)
    
    def on_llm_end(self, response, **kwargs):
        print("=" * 50)
        print("📥 LLM RESPONSE:")
        print("=" * 50)
        for gen in response.generations:
            for g in gen:
                print(g.text)
    
    def on_llm_error(self, error, **kwargs):
        print(f"❌ LLM ERROR: {error}")

# Use
result = chain.invoke(
    dict(input="test"),
    config=dict(callbacks=[PromptDebugHandler()])
)
```

## Inspecting Prompts

### View Formatted Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Explain {topic} to a {audience}")

# See the formatted prompt
formatted = prompt.format(topic="AI", audience="child")
print(formatted)

# See input variables
print(prompt.input_variables)  # ['topic', 'audience']

# See the template
print(prompt.template)
```

### View Message Structure

```python
# For chat prompts
messages = prompt.format_messages(topic="AI", audience="child")
for msg in messages:
    print(f"[{msg.type}] {msg.content}")
```

## Common Issues and Solutions

### Issue 1: Output Format Tidak Konsisten

```text
❌ Problem: Output format varies
prompt = "List 3 colors"
# Output bisa: "1. Red 2. Blue 3. Green"
# Atau: "Red, Blue, Green"

✅ Solution: Be explicit about format
prompt = """
List exactly 3 colors.
Format: One color per line, no numbering or bullets.

Colors:
"""
```

### Issue 2: LLM Ignores Instructions

```text
❌ Problem: Instructions buried in long text

✅ Solution: Put instructions FIRST
"IMPORTANT: Respond ONLY with valid JSON.

Context about colors:
[context here]

Provide response as JSON:"
```

### Issue 3: Inconsistent Language

```text
❌ Problem: Mixes languages
prompt = "Jawab dalam Bahasa Indonesia: What is Python?"

✅ Solution: Clear language instruction
"Pertanyaan: What is Python?

INSTRUKSI: Jawab HANYA dalam Bahasa Indonesia.
Jangan gunakan Bahasa Inggris sama sekali."
```

### Issue 4: Too Creative / Too Rigid

```python
# Use temperature to control
# For factual tasks
llm = ChatOpenAI(temperature=0)

# For creative tasks
llm = ChatOpenAI(temperature=0.8)
```

## Iterative Debugging Process

```python
def debug_prompt(prompt_template, test_input, llm, n=3):
    """Debug a prompt with multiple runs."""
    chain = prompt_template | llm | StrOutputParser()
    
    print(f"Testing prompt with input: {test_input}")
    print("-" * 50)
    
    results = []
    for i in range(n):
        result = chain.invoke(test_input)
        results.append(result)
        print(f"\n--- Run {i+1} ---")
        print(result[:200] + "..." if len(result) > 200 else result)
    
    # Check consistency
    unique_results = set(results)
    consistency = 1 - (len(unique_results) - 1) / n
    print(f"\n📊 Consistency: {consistency:.0%}")
    
    return results

# Use
results = debug_prompt(
    my_prompt,
    dict(topic="Python"),
    ChatOpenAI(model="gpt-4o-mini"),
    n=5
)
```

## A/B Testing Prompts

```python
import random
from typing import Callable

def ab_test_prompts(
    prompt_a,
    prompt_b,
    test_inputs: list,
    llm,
    evaluator: Callable
) -> dict:
    """A/B test two prompts."""
    results_a = []
    results_b = []
    
    for test_input in test_inputs:
        # Test prompt A
        result_a = (prompt_a | llm | StrOutputParser()).invoke(test_input)
        score_a = evaluator(result_a)
        results_a.append(score_a)
        
        # Test prompt B
        result_b = (prompt_b | llm | StrOutputParser()).invoke(test_input)
        score_b = evaluator(result_b)
        results_b.append(score_b)
    
    return dict(
        prompt_a_avg=sum(results_a) / len(results_a),
        prompt_b_avg=sum(results_b) / len(results_b)
    )

# Example evaluator (length-based)
def length_evaluator(response: str) -> float:
    return min(len(response) / 500, 1.0)

# Test
result = ab_test_prompts(
    prompt_v1,
    prompt_v2,
    [dict(topic="AI"), dict(topic="ML")],
    llm,
    length_evaluator
)
print(f"Prompt A: {result['prompt_a_avg']:.2f}")
print(f"Prompt B: {result['prompt_b_avg']:.2f}")
```

## LangSmith Tracing

Best tool untuk debugging di production.

```python
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_sk_..."
os.environ["LANGCHAIN_PROJECT"] = "prompt-debugging"

# All chain runs will be traced
result = chain.invoke(dict(input="test"))

# View at: smith.langchain.com
```

### What LangSmith Shows:

- Full prompt sent to LLM
- Complete response
- Token usage and cost
- Latency breakdown
- Error details

## Systematic Improvement

### 1. Collect Failures

```python
failures = []

def track_failures(chain, test_cases, expected_format):
    for test in test_cases:
        try:
            result = chain.invoke(test)
            if not expected_format(result):
                failures.append(dict(
                    input=test,
                    output=result,
                    reason="format_mismatch"
                ))
        except Exception as e:
            failures.append(dict(
                input=test,
                error=str(e),
                reason="exception"
            ))
    
    return failures
```

### 2. Analyze Patterns

```python
# Group failures by reason
from collections import Counter

failure_reasons = Counter(f["reason"] for f in failures)
print(failure_reasons)
```

### 3. Iterate

```python
# Based on patterns, update prompt
if "format_mismatch" in failure_reasons:
    # Add more explicit format instructions
    updated_prompt = add_format_examples(original_prompt)
```

## Debugging Checklist

✅ **Check formatted prompt** - Is it what you expect?
✅ **Check temperature** - Too high = inconsistent
✅ **Check token limits** - Is output truncated?
✅ **Check edge cases** - Empty input, long input, special chars
✅ **Check consistency** - Multiple runs produce similar output?
✅ **Check language** - Response in expected language?
✅ **Use verbose mode** - See what's happening
✅ **Use LangSmith** - Production debugging

## Ringkasan

1. **set_verbose(True)** untuk quick debugging
2. **Callbacks** untuk custom logging
3. **Inspect prompts** sebelum send ke LLM
4. **Iterative testing** dengan multiple runs
5. **A/B testing** untuk compare prompts
6. **LangSmith** untuk production tracing
7. **Systematic improvement** - collect, analyze, iterate

---

## 🎯 Use Case Modul 3: Code Reviewer Bot

```python
#!/usr/bin/env python3
"""
Code Reviewer Bot - Use Case Modul 3
Few-shot prompting untuk code review dengan style guide.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# Few-shot examples (simplified for MDX compatibility)
examples = [
    dict(
        code="def calc(x,y):\n    return x+y",
        review="""## Code Review
### Issues
1. **Naming** (Warning): Function name too vague
2. **Type hints** (Info): Missing type annotations
### Improved Code
Use: def add_numbers(x: int, y: int) -> int"""
    ),
    dict(
        code="def get_user(id):\n    for u in users:\n        if u.id == id: return u",
        review="""## Code Review
### Issues
1. **Performance** (Warning): O(n) lookup, use dict
2. **Naming** (Info): id shadows built-in
### Improved Code
Use dictionary lookup for O(1) access"""
    )
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Review this code:\n{code}"),
    ("ai", "{review}")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Python code reviewer.
Focus on: code quality, performance, PEP 8, type hints.
Rate issues as: Critical, Warning, or Info."""),
    few_shot,
    ("human", "Review this code:\n{code}")
])

chain = review_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

def review_code(code: str) -> str:
    """Review Python code and provide feedback."""
    return chain.invoke(dict(code=code))

if __name__ == "__main__":
    test_code = '''
def process(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
'''
    
    print("🔍 Code Review Bot\n")
    print("Input Code:")
    print(test_code)
    print("\n" + "="*50 + "\n")
    print(review_code(test_code))
```

**Selamat!** 🎉 Kamu sudah menyelesaikan Modul 3: Prompt Engineering!

---

**Selanjutnya:** [Modul 4: Structured Output](/docs/structured-output/mengapa-structured-output) - Memaksa LLM menghasilkan data terstruktur.
