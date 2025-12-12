---
sidebar_position: 4
title: Branching & Routing
description: Conditional execution dan dynamic routing dalam LCEL chains
---

# Branching & Routing

Tidak semua workflows linear. Kadang kita perlu **branching** (cabang kondisional) atau **routing** (arahkan ke chain berbeda). LCEL menyediakan tools untuk ini.

## RunnableBranch

Eksekusi kondisional berdasarkan input.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Define specialized chains
math_chain = (
    ChatPromptTemplate.from_template("Solve this math problem: {input}")
    | llm
    | StrOutputParser()
)

code_chain = (
    ChatPromptTemplate.from_template("Write code for: {input}")
    | llm
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("Answer: {input}")
    | llm
    | StrOutputParser()
)

# Create branch
branch = RunnableBranch(
    # (condition, chain) tuples
    (lambda x: "math" in x["input"].lower(), math_chain),
    (lambda x: "code" in x["input"].lower(), code_chain),
    # Default (no condition)
    general_chain
)

# Test
print(branch.invoke({"input": "What is 2 + 2?"}))  # Uses math_chain
print(branch.invoke({"input": "Write code for sorting"}))  # Uses code_chain
print(branch.invoke({"input": "What is the capital of Japan?"}))  # Uses general_chain
```

### Branch dengan Classification

Lebih robust - gunakan LLM untuk classify input:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Classifier chain
classifier = (
    ChatPromptTemplate.from_template("""
    Classify this query into one of: math, code, general
    
    Query: {input}
    
    Classification (one word only):
    """)
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
    | RunnableLambda(lambda x: x.strip().lower())
)

# Branch based on classification
def route_by_classification(info: dict):
    classification = info["classification"]
    
    if classification == "math":
        return math_chain
    elif classification == "code":
        return code_chain
    else:
        return general_chain

# Full pipeline
full_chain = (
    # Add classification to input
    {"input": lambda x: x["input"], "classification": classifier}
    # Route
    | RunnableLambda(route_by_classification)
)
```

## Dynamic Routing dengan RunnableLambda

Lebih flexible - gunakan function untuk menentukan chain.

```python
from langchain_core.runnables import RunnableLambda

def route_question(input_dict: dict):
    """Route to appropriate chain based on question type."""
    question = input_dict["question"].lower()
    
    if any(word in question for word in ["calculate", "compute", "math", "sum"]):
        return math_chain
    elif any(word in question for word in ["code", "program", "function", "script"]):
        return code_chain
    elif any(word in question for word in ["translate", "bahasa"]):
        return translate_chain
    else:
        return general_chain

# Create router
router = RunnableLambda(route_question)

# Use router
chain = {"question": lambda x: x["question"]} | router

result = chain.invoke({"question": "Calculate the factorial of 5"})
```

## Router Chain Pattern

Pattern yang lebih terstruktur untuk routing.

```python
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

# Define route schema
class Route(BaseModel):
    """Route classification."""
    destination: Literal["math", "code", "translate", "general"] = Field(
        description="The type of question"
    )
    confidence: float = Field(description="Confidence 0-1")

# Router using structured output
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Route)

router_chain = (
    ChatPromptTemplate.from_template("""
    Classify this question into a category.
    
    Question: {question}
    """)
    | router_llm
)

# Route mapping
route_map = {
    "math": math_chain,
    "code": code_chain,
    "translate": translate_chain,
    "general": general_chain
}

def execute_route(input_dict: dict):
    route = input_dict["route"]
    question = input_dict["question"]
    
    selected_chain = route_map.get(route.destination, general_chain)
    return selected_chain.invoke({"input": question})

# Full routing chain
full_router = (
    {"question": lambda x: x["question"], "route": router_chain}
    | RunnableLambda(execute_route)
)

result = full_router.invoke({"question": "What is 15 * 23?"})
```

## Conditional Execution

### Skip Steps Conditionally

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def maybe_translate(input_dict: dict):
    """Only translate if not already in English."""
    if input_dict.get("language") == "en":
        return input_dict["text"]  # Skip translation
    else:
        return translate_chain.invoke({"text": input_dict["text"]})

chain = (
    {"text": lambda x: x["text"], "language": detect_language}
    | RunnableLambda(maybe_translate)
)
```

### Early Exit

```python
def check_cache(input_dict: dict):
    """Return cached result if available."""
    cached = get_from_cache(input_dict["query"])
    if cached:
        return {"result": cached, "cached": True}
    return {"result": None, "cached": False, **input_dict}

def maybe_generate(input_dict: dict):
    """Generate only if not cached."""
    if input_dict["cached"]:
        return input_dict["result"]
    
    result = generation_chain.invoke(input_dict)
    save_to_cache(input_dict["query"], result)
    return result

chain = (
    RunnableLambda(check_cache)
    | RunnableLambda(maybe_generate)
)
```

## Multi-path Processing

Proses input melalui multiple paths dan combine.

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Process same input through different "lenses"
multi_analysis = RunnableParallel(
    sentiment=sentiment_chain,
    topics=topic_extraction_chain,
    entities=entity_extraction_chain,
    summary=summary_chain
)

# Combine results
def synthesize(analyses: dict) -> str:
    return f"""
    Analysis Report:
    ================
    Sentiment: {analyses['sentiment']}
    Topics: {analyses['topics']}
    Entities: {analyses['entities']}
    
    Summary: {analyses['summary']}
    """

full_chain = multi_analysis | RunnableLambda(synthesize)
```

## Practical Example: Smart Assistant

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

llm = ChatOpenAI(model="gpt-4o-mini")

# Intent classification
class Intent(BaseModel):
    intent: Literal["question", "task", "conversation", "code"] = Field(
        description="User intent type"
    )

classifier = llm.with_structured_output(Intent)

classify_chain = (
    ChatPromptTemplate.from_template("Classify user intent: {input}")
    | classifier
)

# Specialized handlers
question_handler = (
    ChatPromptTemplate.from_template("""
    Answer this question accurately and concisely.
    Question: {input}
    """)
    | llm
    | StrOutputParser()
)

task_handler = (
    ChatPromptTemplate.from_template("""
    Help complete this task. Provide step-by-step guidance.
    Task: {input}
    """)
    | llm
    | StrOutputParser()
)

conversation_handler = (
    ChatPromptTemplate.from_template("""
    Respond naturally to this message in a friendly way.
    Message: {input}
    """)
    | llm
    | StrOutputParser()
)

code_handler = (
    ChatPromptTemplate.from_template("""
    Write clean, well-commented code for this request.
    Request: {input}
    """)
    | llm
    | StrOutputParser()
)

# Router
def route_by_intent(data: dict):
    intent = data["intent"].intent
    handlers = {
        "question": question_handler,
        "task": task_handler,
        "conversation": conversation_handler,
        "code": code_handler
    }
    return handlers.get(intent, conversation_handler).invoke({"input": data["input"]})

# Full smart assistant
smart_assistant = (
    RunnableParallel(
        input=lambda x: x["input"],
        intent=classify_chain
    )
    | RunnableLambda(route_by_intent)
)

# Test
print(smart_assistant.invoke({"input": "What is the capital of France?"}))
print(smart_assistant.invoke({"input": "Write a Python function to sort a list"}))
print(smart_assistant.invoke({"input": "Hey, how are you doing?"}))
```

## Fallback Routes

Handle errors dengan fallback chains.

```python
from langchain_core.runnables import RunnableLambda

def with_error_fallback(main_chain, fallback_chain):
    """Wrap chain with error handling."""
    def execute(input_data):
        try:
            return main_chain.invoke(input_data)
        except Exception as e:
            print(f"Main chain failed: {e}, using fallback")
            return fallback_chain.invoke(input_data)
    
    return RunnableLambda(execute)

# Use
safe_chain = with_error_fallback(
    main_chain=specialized_chain,
    fallback_chain=general_chain
)
```

## Ringkasan

1. **RunnableBranch** - kondisional dengan (condition, chain) tuples
2. **RunnableLambda routing** - custom routing logic
3. **Router Chain Pattern** - LLM-based classification + routing
4. **Conditional execution** - skip steps based on conditions
5. **Multi-path processing** - RunnableParallel untuk multiple analyses
6. **Fallback routes** - handle errors gracefully

---

**Selanjutnya:** [Error Handling](/docs/lcel/error-handling) - Retry, fallbacks, dan error recovery dalam LCEL.
