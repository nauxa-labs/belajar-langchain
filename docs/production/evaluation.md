---
sidebar_position: 3
title: Evaluation
description: Testing dan evaluating LLM outputs
---

# Evaluation

Evaluation memastikan LLM application **bekerja dengan benar** dan **tidak regress**.

## Why Evaluate?

```
Without Evaluation:
"It seems to work" → Push to prod → Users report issues

With Evaluation:
Test suite → Measure quality → Catch regressions → Deploy confidently
```

## LangSmith Datasets

### Create Dataset

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset(
    dataset_name="qa-test-cases",
    description="Test cases for Q&A chain"
)

# Add examples
client.create_example(
    dataset_id=dataset.id,
    inputs={"question": "What is Python?"},
    outputs={"answer": "Python is a programming language"}
)

client.create_example(
    dataset_id=dataset.id,
    inputs={"question": "Who created Python?"},
    outputs={"answer": "Guido van Rossum"}
)
```

### Upload from CSV

```python
import pandas as pd

df = pd.read_csv("test_cases.csv")

dataset = client.create_dataset("from-csv")

for _, row in df.iterrows():
    client.create_example(
        dataset_id=dataset.id,
        inputs={"question": row["question"]},
        outputs={"answer": row["expected_answer"]}
    )
```

### From Production Runs

```python
# Filter good runs
runs = client.list_runs(
    project_name="production",
    filter="eq(feedback_score, 1)"  # Thumbs up
)

dataset = client.create_dataset("production-examples")

for run in runs:
    client.create_example(
        dataset_id=dataset.id,
        inputs=run.inputs,
        outputs=run.outputs
    )
```

## Running Evaluations

### Basic Evaluation

```python
from langsmith.evaluation import evaluate

def my_chain(inputs: dict) -> dict:
    """The chain to evaluate."""
    question = inputs["question"]
    answer = llm.invoke(question)
    return {"answer": answer.content}

# Run evaluation
results = evaluate(
    my_chain,
    data="qa-test-cases",  # Dataset name
    evaluators=[],  # We'll add these
    experiment_prefix="v1-baseline"
)
```

### Built-in Evaluators

```python
from langsmith.evaluation import LangChainStringEvaluator

# Exact match
exact_match = LangChainStringEvaluator("exact_match")

# String contains
contains = LangChainStringEvaluator("criteria", config={
    "criteria": "Does the answer contain the expected information?"
})

# LLM-as-judge
correctness = LangChainStringEvaluator("correctness")

results = evaluate(
    my_chain,
    data="qa-test-cases",
    evaluators=[exact_match, correctness]
)
```

### Custom Evaluators

```python
from langsmith.evaluation import RunEvaluator

def length_evaluator(run, example) -> dict:
    """Check if answer is appropriate length."""
    answer = run.outputs.get("answer", "")
    is_valid = 10 < len(answer) < 1000
    
    return {
        "key": "valid_length",
        "score": 1 if is_valid else 0,
        "comment": f"Length: {len(answer)}"
    }

def contains_keywords(run, example) -> dict:
    """Check for required keywords."""
    answer = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    
    # Extract key terms from expected
    key_terms = expected.split()[:3]  # First 3 words
    matches = sum(1 for term in key_terms if term in answer)
    score = matches / len(key_terms)
    
    return {
        "key": "keyword_overlap",
        "score": score
    }

results = evaluate(
    my_chain,
    data="qa-test-cases",
    evaluators=[length_evaluator, contains_keywords]
)
```

### LLM-as-Judge

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

evaluator_llm = ChatOpenAI(model="gpt-4o-mini")

def llm_judge(run, example) -> dict:
    """Use LLM to judge correctness."""
    
    question = run.inputs.get("question")
    actual = run.outputs.get("answer")
    expected = example.outputs.get("answer")
    
    prompt = ChatPromptTemplate.from_template("""
    Question: {question}
    
    Expected Answer: {expected}
    
    Actual Answer: {actual}
    
    Rate the actual answer on a scale of 1-5:
    1 = Completely wrong
    3 = Partially correct
    5 = Fully correct
    
    Reply with just the number.
    """)
    
    chain = prompt | evaluator_llm
    result = chain.invoke({
        "question": question,
        "expected": expected,
        "actual": actual
    })
    
    score = int(result.content.strip()) / 5
    
    return {
        "key": "llm_correctness",
        "score": score
    }

results = evaluate(
    my_chain,
    data="qa-test-cases",
    evaluators=[llm_judge]
)
```

## RAG Evaluators

### Retrieval Quality

```python
def retrieval_evaluator(run, example) -> dict:
    """Evaluate retrieved documents."""
    
    # Assume chain returns retrieved docs
    docs = run.outputs.get("context", [])
    expected_source = example.outputs.get("source")
    
    # Check if expected source in retrieved
    sources = [d.metadata.get("source") for d in docs]
    found = expected_source in sources
    
    return {
        "key": "retrieval_hit",
        "score": 1 if found else 0
    }
```

### Answer Faithfulness

```python
def faithfulness_evaluator(run, example) -> dict:
    """Check if answer is grounded in context."""
    
    context = "\n".join(run.outputs.get("context", []))
    answer = run.outputs.get("answer", "")
    
    prompt = f"""
    Context: {context}
    
    Answer: {answer}
    
    Is the answer fully supported by the context? (yes/no)
    """
    
    result = evaluator_llm.invoke(prompt)
    is_faithful = "yes" in result.content.lower()
    
    return {
        "key": "faithfulness",
        "score": 1 if is_faithful else 0
    }
```

## Regression Testing

### Compare Experiments

```python
# Run baseline
results_v1 = evaluate(
    chain_v1,
    data="test-cases",
    experiment_prefix="baseline"
)

# Run new version
results_v2 = evaluate(
    chain_v2,
    data="test-cases",
    experiment_prefix="new-model"
)

# Compare in LangSmith dashboard
# Or programmatically:
print(f"V1 avg score: {results_v1.aggregate_scores}")
print(f"V2 avg score: {results_v2.aggregate_scores}")
```

### CI/CD Integration

```python
# tests/test_llm_quality.py
import pytest
from langsmith.evaluation import evaluate

def test_qa_quality():
    """Ensure QA chain meets quality threshold."""
    
    results = evaluate(
        qa_chain,
        data="regression-test-cases",
        evaluators=[correctness_evaluator]
    )
    
    avg_score = results.aggregate_scores["correctness"]
    assert avg_score >= 0.8, f"Quality dropped: {avg_score}"

def test_latency():
    """Ensure response time is acceptable."""
    
    results = evaluate(
        qa_chain,
        data="regression-test-cases"
    )
    
    avg_latency = sum(r.total_time for r in results) / len(results)
    assert avg_latency < 5.0, f"Too slow: {avg_latency}s"
```

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: LLM Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run LLM tests
        env:
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/test_llm_quality.py
```

## Batch Evaluation

### Large Scale

```python
from langsmith.evaluation import evaluate

# Evaluate on large dataset
results = evaluate(
    my_chain,
    data="large-dataset",  # 1000+ examples
    max_concurrency=10,  # Parallel execution
    evaluators=[evaluator]
)

# Monitor progress in LangSmith
```

### Async Evaluation

```python
import asyncio

async def async_evaluate():
    results = await aevaluate(
        async_chain,
        data="test-cases",
        evaluators=[evaluator]
    )
    return results

results = asyncio.run(async_evaluate())
```

## Ringkasan

1. **Datasets** - curated test cases
2. **Built-in evaluators** - exact match, LLM-as-judge
3. **Custom evaluators** - domain-specific checks
4. **RAG evaluators** - retrieval + faithfulness
5. **Regression tests** - CI/CD integration
6. **Batch evaluation** - large scale testing

---

**Selanjutnya:** [LangServe](/docs/production/langserve) - Deploy chain sebagai REST API.
