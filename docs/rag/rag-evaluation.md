---
sidebar_position: 9
title: RAG Evaluation
description: Mengukur dan meningkatkan kualitas RAG system
---

# RAG Evaluation

Membangun RAG baru setengah perjuangan - kita perlu **mengukur** dan **meningkatkan** kualitasnya secara sistematis.

## Evaluation Dimensions

RAG memiliki dua komponen utama yang perlu dievaluasi:

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Evaluation                        │
├───────────────────────┬─────────────────────────────────┤
│   Retrieval Quality   │      Generation Quality         │
├───────────────────────┼─────────────────────────────────┤
│ • Context Relevance   │ • Faithfulness                  │
│ • Context Recall      │ • Answer Relevance              │
│ • Context Precision   │ • Answer Correctness            │
└───────────────────────┴─────────────────────────────────┘
```

## Key Metrics

### 1. Context Relevance

Apakah retrieved documents relevan dengan pertanyaan?

```python
def evaluate_context_relevance(question: str, contexts: list, llm) -> float:
    """Score 0-1: How relevant are the contexts to the question."""
    
    prompt = f"""
    Rate the relevance of these documents to the question.
    Score from 0 (not relevant) to 10 (highly relevant).
    
    Question: {question}
    
    Documents:
    {chr(10).join(contexts)}
    
    Score (0-10):
    """
    
    score = llm.invoke(prompt).content
    return float(score.strip()) / 10
```

### 2. Context Recall

Apakah semua informasi yang dibutuhkan berhasil di-retrieve?

```python
def evaluate_context_recall(question: str, ground_truth: str, contexts: list, llm) -> float:
    """Score 0-1: How much of the ground truth is covered by contexts."""
    
    prompt = f"""
    Given the ground truth answer and retrieved contexts, 
    what percentage of the ground truth information is present in the contexts?
    
    Question: {question}
    Ground Truth: {ground_truth}
    
    Retrieved Contexts:
    {chr(10).join(contexts)}
    
    Coverage percentage (0-100):
    """
    
    score = llm.invoke(prompt).content
    return float(score.strip()) / 100
```

### 3. Faithfulness

Apakah jawaban sesuai dengan context (tidak hallucinate)?

```python
def evaluate_faithfulness(answer: str, contexts: list, llm) -> float:
    """Score 0-1: Is the answer faithful to the contexts (no hallucination)."""
    
    prompt = f"""
    Evaluate if this answer is fully supported by the given contexts.
    Score 0 if answer contains unsupported claims.
    Score 10 if everything in the answer comes from contexts.
    
    Contexts:
    {chr(10).join(contexts)}
    
    Answer: {answer}
    
    Faithfulness score (0-10):
    """
    
    score = llm.invoke(prompt).content
    return float(score.strip()) / 10
```

### 4. Answer Relevance

Apakah jawaban menjawab pertanyaan?

```python
def evaluate_answer_relevance(question: str, answer: str, llm) -> float:
    """Score 0-1: Does the answer address the question."""
    
    prompt = f"""
    Rate how well this answer addresses the question.
    
    Question: {question}
    Answer: {answer}
    
    Relevance score (0-10):
    """
    
    score = llm.invoke(prompt).content
    return float(score.strip()) / 10
```

## Using RAGAS

RAGAS adalah library populer untuk RAG evaluation.

```bash
pip install ragas
```

### Basic Usage

```python
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the vacation policy?",
        "How many sick days do employees get?"
    ],
    "answer": [
        "Employees get 21 days of vacation per year.",
        "Employees receive 10 sick days annually."
    ],
    "contexts": [
        ["According to policy, full-time employees receive 21 days..."],
        ["The company provides 10 sick days per calendar year..."]
    ],
    "ground_truth": [
        "Full-time employees get 21 vacation days per year.",
        "10 sick days are provided to each employee."
    ]
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset,
    metrics=[
        context_relevancy,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)

print(results)
```

### Interpreting Results

```python
"""
{
    'context_relevancy': 0.89,    # Good: contexts are relevant
    'context_recall': 0.75,       # Okay: missing some ground truth info
    'faithfulness': 0.95,         # Great: minimal hallucination
    'answer_relevancy': 0.82      # Good: answers address questions
}
"""
```

## LangSmith Evaluation

Integrate dengan LangSmith untuk tracing + evaluation.

```python
import os
from langsmith import Client
from langsmith.evaluation import evaluate

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_sk_..."

client = Client()

# Create dataset
dataset_name = "rag-evaluation"
dataset = client.create_dataset(dataset_name)

# Add examples
examples = [
    {
        "input": {"question": "What is vacation policy?"},
        "output": {"answer": "21 days per year for full-time."}
    },
    # More examples...
]

for ex in examples:
    client.create_example(
        inputs=ex["input"],
        outputs=ex["output"],
        dataset_id=dataset.id
    )

# Define evaluator
def relevance_evaluator(run, example):
    prediction = run.outputs["answer"]
    expected = example.outputs["answer"]
    
    # Simple check - in production use LLM
    score = 1.0 if any(word in prediction for word in expected.split()) else 0.0
    
    return {"key": "relevance", "score": score}

# Run evaluation
results = evaluate(
    rag_chain.invoke,
    data=dataset_name,
    evaluators=[relevance_evaluator]
)
```

## Creating Evaluation Dataset

### From Actual Documents

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_qa_pairs(document: str, n: int = 3) -> list:
    """Generate question-answer pairs from document."""
    
    prompt = ChatPromptTemplate.from_template("""
    Generate {n} question-answer pairs from this document.
    Questions should be answerable from the document.
    
    Document:
    {document}
    
    Format each as:
    Q: [question]
    A: [answer]
    """)
    
    response = (prompt | llm).invoke({"document": document, "n": n})
    
    # Parse response
    pairs = []
    lines = response.content.strip().split("\n")
    for i in range(0, len(lines), 2):
        if lines[i].startswith("Q:") and i+1 < len(lines):
            q = lines[i][2:].strip()
            a = lines[i+1][2:].strip()
            pairs.append({"question": q, "answer": a})
    
    return pairs

# Generate for all documents
eval_pairs = []
for doc in documents[:10]:  # Sample
    pairs = generate_qa_pairs(doc.page_content)
    for p in pairs:
        p["source"] = doc.metadata.get("source", "unknown")
    eval_pairs.extend(pairs)
```

### Human Annotation

```python
import json

# Export for human review
def export_for_annotation(eval_pairs, output_file):
    with open(output_file, "w") as f:
        json.dump(eval_pairs, f, indent=2)
    print(f"Exported {len(eval_pairs)} pairs to {output_file}")

# After human review, load back
def load_annotated(input_file):
    with open(input_file) as f:
        return json.load(f)
```

## Automated Testing Pipeline

```python
from typing import Dict, List
import json

class RAGEvaluator:
    def __init__(self, rag_chain, llm):
        self.rag_chain = rag_chain
        self.llm = llm
    
    def run_evaluation(self, test_cases: List[Dict]) -> Dict:
        """Run full evaluation on test cases."""
        
        results = {
            "total": len(test_cases),
            "scores": {
                "faithfulness": [],
                "relevance": [],
                "context_quality": []
            },
            "details": []
        }
        
        for case in test_cases:
            question = case["question"]
            expected = case.get("expected_answer", "")
            
            # Get RAG response
            response = self.rag_chain.invoke(question)
            answer = response.get("answer", response)
            contexts = response.get("context", [])
            
            # Evaluate
            faithfulness = self._evaluate_faithfulness(answer, contexts)
            relevance = self._evaluate_relevance(question, answer)
            
            results["scores"]["faithfulness"].append(faithfulness)
            results["scores"]["relevance"].append(relevance)
            
            results["details"].append({
                "question": question,
                "answer": answer,
                "expected": expected,
                "faithfulness": faithfulness,
                "relevance": relevance
            })
        
        # Calculate averages
        for key in results["scores"]:
            scores = results["scores"][key]
            if scores:
                results["scores"][key] = sum(scores) / len(scores)
        
        return results
    
    def _evaluate_faithfulness(self, answer, contexts) -> float:
        # Implementation...
        pass
    
    def _evaluate_relevance(self, question, answer) -> float:
        # Implementation...
        pass

# Usage
evaluator = RAGEvaluator(rag_chain, llm)
results = evaluator.run_evaluation(test_cases)

print(f"Faithfulness: {results['scores']['faithfulness']:.2%}")
print(f"Relevance: {results['scores']['relevance']:.2%}")
```

## Debugging Poor Performance

### Low Context Relevance

**Symptoms:** Retrieved documents not related to query.

**Solutions:**
1. Improve embedding model
2. Add hybrid search
3. Tune chunk size
4. Add query expansion

```python
# Debug: View what's being retrieved
def debug_retrieval(query, retriever, k=5):
    docs = retriever.invoke(query)
    
    print(f"Query: {query}\n")
    for i, doc in enumerate(docs):
        print(f"--- Doc {i+1} ---")
        print(f"Score: {getattr(doc, 'score', 'N/A')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(doc.page_content[:300])
        print()
```

### Low Faithfulness (Hallucination)

**Symptoms:** Answer contains info not in context.

**Solutions:**
1. Improve prompt (be stricter)
2. Lower temperature
3. Add "if not found, say so" instruction

```python
# Stricter prompt
strict_prompt = """
Answer ONLY based on the provided context.
If the information is not in the context, respond:
"I cannot find this information in the available documents."

Context:
{context}

Question: {question}

Answer:
"""
```

### Low Answer Relevance

**Symptoms:** Answer doesn't address the question.

**Solutions:**
1. Improve prompt clarity
2. Use better LLM
3. Add examples in prompt

## Continuous Evaluation

```python
import logging
from datetime import datetime

# Log all RAG interactions for later analysis
def log_rag_interaction(question, answer, contexts, metadata=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "context_count": len(contexts),
        "metadata": metadata or {}
    }
    
    logging.info(json.dumps(log_entry))
    
    # Store for batch evaluation later
    with open("rag_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

## Ringkasan

1. **Key metrics**: Context relevance, faithfulness, answer relevance
2. **RAGAS** - popular evaluation library
3. **LangSmith** - integrated tracing + evaluation
4. Generate **test datasets** from documents
5. **Automate** evaluation in CI/CD
6. **Debug** systematically based on metrics

---

**Selanjutnya:** [RAG Best Practices](/docs/rag/rag-best-practices) - Tips dan patterns untuk production RAG.
