---
sidebar_position: 6
title: Agent Patterns
description: ReAct, Plan-and-Execute, dan patterns lainnya
---

# Agent Patterns

Berbagai pattern untuk membangun agents dengan kemampuan berbeda.

## 1. ReAct Pattern

**Reasoning + Acting** - pattern paling populer.

```
Thought → Action → Observation → Thought → ... → Final Answer
```

### Characteristics

- ✅ Simple, well-tested
- ✅ Good for most use cases
- ❌ Can get stuck in loops
- ❌ May not plan ahead

### Implementation

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ReAct-style prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that thinks step by step.

For each problem:
1. Think about what you need to do
2. Use tools to gather information or perform actions
3. Analyze the results
4. Continue until you have a complete answer

Always explain your reasoning."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

## 2. Plan-and-Execute Pattern

**Plan first, then execute** - better for complex multi-step tasks.

```
Input → [Planner] → Plan → [Executor] → Step 1 → Step 2 → ... → Output
```

### Characteristics

- ✅ Better for complex tasks
- ✅ More structured approach
- ❌ Planning takes time
- ❌ Less flexible to changes

### Implementation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class Plan(BaseModel):
    """A plan for solving a task."""
    steps: List[str] = Field(description="Steps to complete the task")
    
# Planner
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a planning assistant. 
Given a task, create a step-by-step plan to accomplish it.
Each step should be specific and actionable."""),
    ("human", "Task: {input}\n\nCreate a plan:")
])

planner = planner_prompt | ChatOpenAI(model="gpt-4o-mini").with_structured_output(Plan)

# Executor (uses tools)
executor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are executing a plan.
Current step: {current_step}
Previous results: {previous_results}

Execute this step using available tools."""),
    ("human", "Execute the current step"),
    ("placeholder", "{agent_scratchpad}")
])

async def plan_and_execute(task: str):
    # Step 1: Create plan
    plan = await planner.ainvoke({"input": task})
    print(f"Plan: {plan.steps}")
    
    results = []
    
    # Step 2: Execute each step
    for i, step in enumerate(plan.steps):
        print(f"\nExecuting step {i+1}: {step}")
        
        result = await executor.ainvoke({
            "current_step": step,
            "previous_results": "\n".join(results)
        })
        
        results.append(f"Step {i+1}: {result['output']}")
    
    return results
```

## 3. Self-Ask with Search

Agent asks itself clarifying questions and searches for answers.

```
Question → Sub-question 1 → Search → Answer 1 → Sub-question 2 → ...
```

### Implementation

```python
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

@tool
def search(query: str) -> str:
    """Search for factual information."""
    # Implementation
    return f"Search result for: {query}"

self_ask_prompt = ChatPromptTemplate.from_messages([
    ("system", """You answer questions by breaking them into sub-questions.

Process:
1. Identify what you need to know to answer
2. Ask a simpler follow-up question
3. Use search to find the answer
4. Repeat until you can answer the original question

Format:
Follow up: <simpler question>
Intermediate answer: <answer from search>
...
So the final answer is: <final answer>"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
```

## 4. Reflexion Pattern

Agent **reflects** on its actions and improves.

```
Action → Result → Reflection → Improved Action → ...
```

### Implementation

```python
from langchain_core.prompts import ChatPromptTemplate

reflection_prompt = ChatPromptTemplate.from_template("""
You attempted to solve a problem but the result was not satisfactory.

Original task: {task}
Your attempt: {attempt}
Result: {result}
Feedback: {feedback}

Reflect on what went wrong and how to improve:
1. What was the issue?
2. What should you do differently?
3. Create an improved approach.

Reflection:
""")

async def reflexion_loop(task: str, max_attempts: int = 3):
    attempt = None
    
    for i in range(max_attempts):
        # Make attempt
        result = await executor.ainvoke({"input": task})
        
        # Check if satisfactory
        is_good = await evaluate_result(result["output"])
        
        if is_good:
            return result["output"]
        
        # Reflect and improve
        reflection = await reflection_prompt.ainvoke({
            "task": task,
            "attempt": result["output"],
            "result": "Not satisfactory",
            "feedback": "Try a different approach"
        })
        
        # Use reflection to improve next attempt
        task = f"{task}\n\nPrevious reflection: {reflection}"
    
    return result["output"]
```

## 5. Tool-Selecting Agent

Agent first decides which tools to use, then uses them.

```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class ToolSelection(BaseModel):
    """Selected tools for a task."""
    tools: List[str] = Field(description="Names of tools to use")
    reasoning: str = Field(description="Why these tools")

selector_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a tool selector.
Given a task and available tools, select which tools are needed.

Available tools:
{tool_descriptions}"""),
    ("human", "Task: {input}\n\nSelect tools:")
])

async def tool_selecting_agent(task: str, all_tools: list):
    # Get tool descriptions
    descriptions = "\n".join([
        f"- {t.name}: {t.description}" for t in all_tools
    ])
    
    # Select tools
    selection = await (
        selector_prompt 
        | llm.with_structured_output(ToolSelection)
    ).ainvoke({
        "input": task,
        "tool_descriptions": descriptions
    })
    
    # Create agent with only selected tools
    selected = [t for t in all_tools if t.name in selection.tools]
    
    agent = create_tool_calling_agent(llm, selected, prompt)
    executor = AgentExecutor(agent=agent, tools=selected)
    
    return await executor.ainvoke({"input": task})
```

## 6. Hierarchical Agents

**Manager agent** delegates to **worker agents**.

```
      ┌─────────┐
      │ Manager │
      └────┬────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
[Worker][Worker][Worker]
```

### Implementation

```python
from langchain_core.tools import tool

# Worker agents as tools
@tool
def research_agent(query: str) -> str:
    """Research agent that finds information."""
    # Run research agent
    result = research_executor.invoke({"input": query})
    return result["output"]

@tool
def writing_agent(topic: str) -> str:
    """Writing agent that creates content."""
    result = writing_executor.invoke({"input": topic})
    return result["output"]

@tool
def review_agent(content: str) -> str:
    """Review agent that checks quality."""
    result = review_executor.invoke({"input": content})
    return result["output"]

# Manager agent
manager_tools = [research_agent, writing_agent, review_agent]

manager_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a manager agent that coordinates worker agents.

Available workers:
- research_agent: Finds information
- writing_agent: Creates content
- review_agent: Reviews and improves content

Delegate tasks appropriately and combine results."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

manager = create_tool_calling_agent(llm, manager_tools, manager_prompt)
manager_executor = AgentExecutor(agent=manager, tools=manager_tools)
```

## Pattern Comparison

| Pattern | Best For | Complexity | Latency |
|---------|----------|------------|---------|
| ReAct | General tasks | Low | Low |
| Plan-and-Execute | Multi-step tasks | Medium | Medium |
| Self-Ask | Factual Q&A | Low | Medium |
| Reflexion | Quality-critical | High | High |
| Tool-Selecting | Many tools | Medium | Low |
| Hierarchical | Complex workflows | High | High |

## Choosing a Pattern

### Use ReAct when:
- Task is straightforward
- Real-time response needed
- Limited number of tools

### Use Plan-and-Execute when:
- Multi-step complex task
- Steps can be predefined
- Quality > Speed

### Use Hierarchical when:
- Many diverse capabilities
- Need specialization
- Complex workflows

## Best Practices

### 1. Clear Role Definitions

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a DATA ANALYST agent.

Your role:
- Analyze data using available tools
- Create visualizations when helpful
- Explain findings clearly

You should NOT:
- Make up data
- Guess when unsure
- Skip verification steps"""),
    ...
])
```

### 2. Iteration Limits

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # Prevent infinite loops
    max_execution_time=60  # Timeout
)
```

### 3. Fallback Strategies

```python
async def run_with_fallback(query: str):
    try:
        return await executor.ainvoke({"input": query})
    except Exception as e:
        # Fallback to simpler approach
        return await llm.ainvoke(f"Answer without tools: {query}")
```

## Ringkasan

1. **ReAct** - Think, Act, Observe loop
2. **Plan-and-Execute** - Plan first, then execute
3. **Self-Ask** - Break into sub-questions
4. **Reflexion** - Learn from mistakes
5. **Hierarchical** - Manager + Workers
6. Choose based on **task complexity** and **latency requirements**

---

**Selanjutnya:** [Streaming Agents](/docs/agents/streaming-agents) - Building responsive agent UIs.
