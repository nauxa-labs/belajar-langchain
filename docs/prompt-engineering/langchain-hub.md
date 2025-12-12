---
sidebar_position: 4
title: LangChain Hub
description: Menggunakan dan sharing prompts melalui LangChain Hub
---

# LangChain Hub

LangChain Hub adalah repository untuk prompts yang bisa digunakan dan di-share. Ini memungkinkan version control dan kolaborasi untuk prompt engineering.

## Apa itu LangChain Hub?

- **Repository** untuk prompts, chains, dan agents
- **Version control** untuk iterasi prompt
- **Sharing** dengan tim atau komunitas
- **Discovery** - temukan prompts yang sudah battle-tested

URL: [smith.langchain.com/hub](https://smith.langchain.com/hub)

## Setup

```bash
pip install langchainhub
```

```python
from langchain import hub
```

## Menggunakan Prompts dari Hub

### Pull Prompt

```python
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Pull prompt dari hub
prompt = hub.pull("hwchase17/react")  # Popular ReAct prompt

print(prompt.template)  # Lihat isi prompt
```

### Contoh Prompts Populer

```python
# ReAct Agent Prompt
react_prompt = hub.pull("hwchase17/react")

# XML Agent Prompt
xml_prompt = hub.pull("hwchase17/xml-agent-convo")

# RAG Prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# Structured Chat Agent
chat_agent_prompt = hub.pull("hwchase17/structured-chat-agent")
```

### Menggunakan dalam Chain

```python
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Pull RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# Create chain
chain = rag_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

# Use
result = chain.invoke({
    "context": "LangChain is a framework for developing applications powered by LLMs.",
    "question": "What is LangChain?"
})
```

## Menyimpan Prompts ke Hub

### Setup API Key

```bash
export LANGCHAIN_API_KEY="lsv2_sk_..."
```

### Push Prompt

```python
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# Create your prompt
my_prompt = ChatPromptTemplate.from_template("""
You are a helpful coding assistant specializing in Python.

User question: {question}

Provide a clear, concise answer with code examples when appropriate.
""")

# Push to hub (requires authentication)
hub.push("my-username/python-coding-assistant", my_prompt)
```

### Versioning

```python
# Push new version
hub.push("my-username/python-coding-assistant", updated_prompt)

# Pull specific version
prompt_v1 = hub.pull("my-username/python-coding-assistant:v1")
prompt_v2 = hub.pull("my-username/python-coding-assistant:v2")
prompt_latest = hub.pull("my-username/python-coding-assistant:latest")
```

## Exploring the Hub

### Browse Categories

Hub memiliki berbagai kategori:

| Category | Description |
|----------|-------------|
| Agents | ReAct, XML, structured agents |
| RAG | Retrieval prompts |
| Chains | General purpose chains |
| Chat | Conversational prompts |
| Summarization | Text summarization |

### Search

```python
# Di web interface: smith.langchain.com/hub
# Search: "rag", "agent", "summarize", etc.
```

## Best Practices

### 1. Version Your Prompts

```python
# Always tag versions for production
hub.push("team/customer-support-prompt", prompt, tags=["v1.0", "production"])
```

### 2. Document Your Prompts

```python
# Good naming
hub.push("team/summarize-for-executives", prompt)  # ✅ Descriptive

# Bad naming
hub.push("team/prompt1", prompt)  # ❌ Not descriptive
```

### 3. Test Before Push

```python
# Test prompt locally first
test_result = chain.invoke({"question": "test input"})
assert len(test_result) > 0  # Basic validation

# Then push
hub.push("team/tested-prompt", prompt)
```

## Local Prompt Management

Untuk tim yang tidak ingin pakai Hub publik.

### File-based Prompts

```python
# prompts/customer_service.py
from langchain_core.prompts import ChatPromptTemplate

CUSTOMER_SERVICE_PROMPT = ChatPromptTemplate.from_template("""
You are a friendly customer service representative for {company_name}.

Guidelines:
- Be helpful and empathetic
- Offer solutions, not excuses
- Escalate complex issues

Customer: {message}
Response:
""")
```

```python
# Usage
from prompts.customer_service import CUSTOMER_SERVICE_PROMPT

chain = CUSTOMER_SERVICE_PROMPT | llm | parser
```

### YAML-based Prompts

```yaml
# prompts/code_review.yaml
template: |
  You are a senior code reviewer.
  
  Review this {language} code for:
  1. Bugs and errors
  2. Performance issues
  3. Code style
  
  Code:
  ```{language}
  {code}
  ```
  
  Provide structured feedback.

input_variables:
  - language
  - code
```

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/code_review.yaml")
```

### JSON-based Prompts

```json
{
  "_type": "prompt",
  "template": "Translate from {source_lang} to {target_lang}:\n\n{text}",
  "input_variables": ["source_lang", "target_lang", "text"]
}
```

```python
prompt = load_prompt("prompts/translate.json")
```

## Team Workflow

### Development → Staging → Production

```python
# Development
hub.pull("team/my-prompt:dev")

# Staging
hub.pull("team/my-prompt:staging")

# Production
hub.pull("team/my-prompt:production")
```

### Promotion

```python
# After testing in dev
dev_prompt = hub.pull("team/my-prompt:dev")

# Promote to staging
hub.push("team/my-prompt", dev_prompt, tags=["staging"])

# After staging validation, promote to production
hub.push("team/my-prompt", dev_prompt, tags=["production"])
```

## Ringkasan

1. **LangChain Hub** = repository untuk prompts
2. **`hub.pull()`** = download prompts
3. **`hub.push()`** = upload prompts
4. **Versioning** dengan tags untuk iteration
5. **Categories** untuk discovery
6. **Local alternatives** dengan files (YAML, JSON, Python)

---

**Selanjutnya:** [Debugging Prompts](/docs/prompt-engineering/debugging-prompts) - Teknik debugging dan iterasi prompts.
