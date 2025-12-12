---
sidebar_position: 3
title: Advanced Techniques
description: Chain-of-Thought, Self-Consistency, dan teknik prompting lanjutan
---

# Advanced Prompting Techniques

Di bab ini kita akan mempelajari teknik prompting lanjutan yang bisa meningkatkan reasoning dan akurasi LLM.

## Chain-of-Thought (CoT)

Meminta LLM untuk menjelaskan langkah-langkah pemikirannya sebelum memberikan jawaban.

### Standard Prompting vs CoT

```python
# ❌ Standard - langsung jawab
prompt = "Roger has 5 balls. He buys 2 more cans with 3 balls each. How many balls does he have?"
# Output: "11" (bisa salah)

# ✅ CoT - thinking step by step
prompt = """
Roger has 5 balls. He buys 2 more cans with 3 balls each. 
How many balls does he have?

Let's think step by step.
"""
# Output:
# "Step 1: Roger starts with 5 balls.
#  Step 2: He buys 2 cans with 3 balls each = 2 × 3 = 6 balls.
#  Step 3: Total = 5 + 6 = 11 balls.
#  Answer: 11"
```

### Implementasi di LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

cot_prompt = ChatPromptTemplate.from_template("""
{problem}

Let's solve this step by step:
1. First, identify what we know
2. Then, determine what we need to find
3. Next, plan the solution approach
4. Finally, calculate and verify

Solution:
""")

cot_chain = cot_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

result = cot_chain.invoke({
    "problem": "A train travels 120 km in 2 hours. How long will it take to travel 300 km?"
})
print(result)
```

### Zero-shot CoT

Cukup tambahkan "Let's think step by step" di akhir prompt.

```python
zero_shot_cot = ChatPromptTemplate.from_template("""
{question}

Let's think step by step.
""")
```

### Few-shot CoT

Berikan contoh reasoning lengkap.

```python
few_shot_cot = ChatPromptTemplate.from_template("""
Q: There are 15 trees in the grove. Workers will plant trees today. 
After they are done, there will be 21 trees. How many trees did they plant?

A: Let's think step by step.
- We start with 15 trees
- We end with 21 trees
- Trees planted = 21 - 15 = 6
Answer: 6 trees

Q: If there are 3 cars in the parking lot and 2 more cars arrive, 
how many cars are in the parking lot?

A: Let's think step by step.
- We start with 3 cars
- 2 more cars arrive
- Total = 3 + 2 = 5
Answer: 5 cars

Q: {question}

A: Let's think step by step.
""")
```

## Self-Consistency

Generate multiple responses dan pilih yang paling konsisten (majority voting).

```python
from collections import Counter
import re

def extract_answer(response: str) -> str:
    """Extract final answer from CoT response."""
    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*(\d+)', response)
    if match:
        return match.group(1)
    return response.strip().split('\n')[-1]

async def self_consistency(chain, input_data, n=5):
    """Run chain multiple times and return majority answer."""
    # Generate multiple responses
    responses = await chain.abatch([input_data] * n)
    
    # Extract answers
    answers = [extract_answer(r) for r in responses]
    
    # Count votes
    vote_counts = Counter(answers)
    
    # Return majority
    majority_answer, count = vote_counts.most_common(1)[0]
    confidence = count / n
    
    return {
        "answer": majority_answer,
        "confidence": confidence,
        "votes": dict(vote_counts)
    }

# Use
result = await self_consistency(cot_chain, {"problem": "What is 123 * 456?"}, n=5)
print(f"Answer: {result['answer']} (confidence: {result['confidence']:.0%})")
```

### Praktis dengan Temperature

```python
# Gunakan temperature > 0 untuk variasi
cot_chain = (
    cot_prompt 
    | ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Variasi
    | StrOutputParser()
)
```

## Structured Chain-of-Thought

Memaksa format reasoning yang terstruktur.

```python
from pydantic import BaseModel, Field
from typing import List

class ReasoningStep(BaseModel):
    step_number: int
    thought: str
    action: str
    result: str

class StructuredCoT(BaseModel):
    problem_understanding: str = Field(description="What the problem is asking")
    steps: List[ReasoningStep] = Field(description="Reasoning steps")
    final_answer: str = Field(description="The final answer")
    confidence: float = Field(description="Confidence 0-1")

structured_cot_prompt = ChatPromptTemplate.from_template("""
Solve this problem with detailed reasoning.

Problem: {problem}

Think through this step-by-step, showing your work clearly.
""")

chain = (
    structured_cot_prompt 
    | ChatOpenAI(model="gpt-4o-mini").with_structured_output(StructuredCoT)
)

result = chain.invoke({"problem": "If a shirt costs $25 and is on 20% sale, what's the final price?"})
print(f"Understanding: {result.problem_understanding}")
for step in result.steps:
    print(f"Step {step.step_number}: {step.thought}")
print(f"Answer: {result.final_answer}")
```

## ReAct (Reasoning + Acting)

Kombinasi reasoning dengan action taking - fundamental untuk agents.

```python
react_prompt = ChatPromptTemplate.from_template("""
Answer the following question by thinking and taking actions.

Question: {question}

Use this format:
Thought: [Your reasoning about what to do]
Action: [The action to take, one of: Search, Calculate, Lookup]
Action Input: [The input for the action]
Observation: [Result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [The answer]

Begin!
""")
```

### ReAct dengan Tools

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

# ReAct agent
agent = create_react_agent(llm, [calculator, search], react_prompt)
executor = AgentExecutor(agent=agent, tools=[calculator, search])

result = executor.invoke({"question": "What is 25% of 180?"})
```

## Decomposition

Memecah masalah kompleks menjadi sub-masalah.

```python
decompose_prompt = ChatPromptTemplate.from_template("""
Break down this complex problem into smaller, manageable sub-problems.

Complex Problem: {problem}

Sub-problems:
1.
2.
3.
...

Then solve each sub-problem to arrive at the final answer.
""")

# Chain for decomposition then solving
from langchain_core.runnables import RunnableLambda

def solve_subproblems(decomposition: str) -> str:
    # Parse and solve each sub-problem
    subproblems = decomposition.split('\n')
    solutions = []
    for sp in subproblems:
        if sp.strip():
            solution = solve_chain.invoke({"subproblem": sp})
            solutions.append(solution)
    return "\n".join(solutions)

full_chain = decompose_prompt | llm | StrOutputParser() | RunnableLambda(solve_subproblems)
```

## Least-to-Most Prompting

Solve dari yang paling mudah ke yang paling sulit.

```python
least_to_most = ChatPromptTemplate.from_template("""
Solve this by starting from simpler cases and building up.

Problem: {problem}

Step 1: Start with the simplest case
Step 2: Add one level of complexity
Step 3: Continue building up
...
Final: Apply to the full problem

Solution:
""")
```

## Contraposititve Prompting

Minta LLM mempertimbangkan alternatif atau counterarguments.

```python
contrapositive_prompt = ChatPromptTemplate.from_template("""
Question: {question}

First, answer the question.
Then, consider: What if the opposite were true?
Finally, synthesize both perspectives for a balanced answer.

Analysis:
""")
```

## Practical Example: Complex Reasoning

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    problem: str
    assumptions: list[str]
    approach: str
    steps: list[str]
    answer: str
    confidence: str
    alternative_approaches: list[str]

complex_reasoning_prompt = ChatPromptTemplate.from_template("""
You are an expert problem solver. Analyze this problem thoroughly.

Problem: {problem}

Provide a complete analysis including:
1. Restate the problem clearly
2. List any assumptions
3. Describe your approach
4. Show step-by-step solution
5. Give the final answer
6. Rate your confidence (high/medium/low)
7. Suggest alternative approaches
""")

chain = (
    complex_reasoning_prompt 
    | ChatOpenAI(model="gpt-4o-mini").with_structured_output(Analysis)
)

result = chain.invoke({
    "problem": """
    A company has 100 employees. 60% work remotely.
    Of those who work remotely, 80% are satisfied with their work-life balance.
    Of those who work in-office, 50% are satisfied.
    What percentage of all employees are satisfied with their work-life balance?
    """
})

print(f"Problem: {result.problem}")
print(f"Approach: {result.approach}")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
```

## Kapan Pakai Teknik Mana?

| Teknik | Use Case | Overhead |
|--------|----------|----------|
| Zero-shot CoT | Quick reasoning | Low |
| Few-shot CoT | Consistent format | Medium |
| Self-consistency | High-stakes decisions | High |
| Structured CoT | Auditable reasoning | Medium |
| ReAct | Tool-using scenarios | Medium |
| Decomposition | Very complex problems | High |

## Ringkasan

1. **Chain-of-Thought** - "Let's think step by step"
2. **Self-Consistency** - Multiple generations + majority vote
3. **Structured CoT** - Pydantic models untuk format reasoning
4. **ReAct** - Reasoning + Actions untuk agents
5. **Decomposition** - Break complex into simple
6. Pilih teknik sesuai **complexity** dan **stakes**

---

**Selanjutnya:** [LangChain Hub](/docs/prompt-engineering/langchain-hub) - Menggunakan dan sharing prompts.
