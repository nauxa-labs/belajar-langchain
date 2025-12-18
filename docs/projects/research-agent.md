---
sidebar_position: 3
title: "Proyek 2: Research Agent"
description: Agent yang bisa search internet dan menyusun laporan
---

# Proyek 2: Multi-Tool Research Agent

Membangun agent yang bisa **search internet**, **baca Wikipedia**, dan **menyusun laporan research** otomatis.

## Requirements

### Fitur Utama
- ✅ Internet search (Tavily/DuckDuckGo)
- ✅ Wikipedia lookup
- ✅ Calculator
- ✅ Report generation
- ✅ Streaming output

### Tech Stack
- LangChain Agents
- Multiple tools
- Streaming dengan astream_events

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Research Agent                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Research Query                                             │
│        │                                                     │
│        ▼                                                     │
│   [Agent Reasoning]                                          │
│        │                                                     │
│   ┌────┴────┬────────────┬──────────────┐                   │
│   ▼         ▼            ▼              ▼                   │
│ 🔍Search  📚Wikipedia  🔢Calculator  📝Report              │
│   │         │            │              │                   │
│   └─────────┴────────────┴──────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│   [Synthesize & Report]                                      │
│        │                                                     │
│        ▼                                                     │
│   Formatted Research Report                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Define Tools

```python
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Web Search
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the internet for current information.
    
    Use for: recent news, current events, up-to-date data.
    """
    return search.invoke(query)

# Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def lookup_wikipedia(topic: str) -> str:
    """Look up factual information from Wikipedia.
    
    Use for: definitions, historical facts, biographies.
    """
    return wikipedia.invoke(topic)

# Calculator
@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions.
    
    Args:
        expression: Math expression like '2+2' or 'sqrt(16)*5'
    """
    import math
    
    allowed = {
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
        'log': math.log, 'exp': math.exp, 'pi': math.pi, 'e': math.e
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Report Writer
@tool
def write_report(topic: str, findings: str) -> str:
    """Write a structured research report.
    
    Args:
        topic: The research topic
        findings: Key findings from research
    """
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""Write a professional research report.

Topic: {topic}

Key Findings:
{findings}

Format:
- Executive Summary
- Key Points
- Detailed Findings
- Conclusion
"""
    
    response = llm.invoke(prompt)
    return response.content

tools = [web_search, lookup_wikipedia, calculate, write_report]
```

### Step 2: Create Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that thoroughly investigates topics.

Research Process:
1. Search the web for current information
2. Look up Wikipedia for factual background
3. Use calculator for any numerical analysis
4. Synthesize findings into a report

Be thorough but efficient. Cite your sources."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)
```

### Step 3: Streaming Interface

```python
async def stream_research(query: str):
    """Stream research process with live updates."""
    
    print(f"\n🔬 Researching: {query}\n")
    print("=" * 50)
    
    tools_used = []
    
    async for event in executor.astream_events(
        {"input": query},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_tool_start":
            tool_name = event["name"]
            print(f"\n🔧 Using: {tool_name}")
            tools_used.append(tool_name)
        
        elif kind == "on_tool_end":
            output = event["data"]["output"]
            print(f"   ✓ Got result ({len(output)} chars)")
        
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
    
    print("\n" + "=" * 50)
    print(f"✅ Research complete. Tools used: {', '.join(set(tools_used))}")
```

## Complete Code

```python
#!/usr/bin/env python3
"""
Multi-Tool Research Agent
Proyek 2 - Modul 10
"""

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# ===== TOOLS =====

search_engine = DuckDuckGoSearchRun()
wiki_api = WikipediaAPIWrapper()

@tool
def web_search(query: str) -> str:
    """Search the internet for current information and news."""
    try:
        return search_engine.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"

@tool
def wikipedia(topic: str) -> str:
    """Look up factual information from Wikipedia encyclopedia."""
    try:
        wiki = WikipediaQueryRun(api_wrapper=wiki_api)
        return wiki.invoke(topic)
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions. Example: '2+2', 'sqrt(16)'"""
    import math
    allowed = {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
               'tan': math.tan, 'log': math.log, 'exp': math.exp,
               'pi': math.pi, 'e': math.e, 'abs': abs, 'pow': pow}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"

tools = [web_search, wikipedia, calculator]


# ===== AGENT =====

def create_research_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research assistant.

Your process:
1. Break down the research question
2. Use web_search for current info
3. Use wikipedia for factual background
4. Use calculator for any math
5. Synthesize into a clear, cited report

Format your final response as a research report with:
- Summary
- Key Findings
- Sources Used
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        handle_parsing_errors=True
    )


# ===== STREAMING =====

async def research(query: str):
    """Run research with streaming output."""
    executor = create_research_agent()
    
    print(f"\n{'='*60}")
    print(f"🔬 RESEARCH: {query}")
    print(f"{'='*60}\n")
    
    tool_count = 0
    
    async for event in executor.astream_events(
        {"input": query},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_tool_start":
            tool_count += 1
            tool_name = event["name"]
            tool_input = str(event["data"].get("input", ""))[:50]
            print(f"🔧 [{tool_count}] {tool_name}: {tool_input}...")
        
        elif kind == "on_tool_end":
            output = event["data"]["output"]
            preview = output[:100].replace("\n", " ")
            print(f"   ✓ Result: {preview}...\n")
        
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"✅ Research complete. Used {tool_count} tool calls.")
    print(f"{'='*60}\n")


# ===== CLI =====

async def main():
    print("\n🤖 Research Agent Ready!")
    print("Enter a research topic or question. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Research: ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            await research(query)
            
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
```

## Usage Examples

```bash
python research_agent.py

# Example queries:
Research: What are the latest developments in quantum computing?
Research: Compare the GDP of Indonesia and Thailand
Research: What is the history of the Python programming language?
```

## Improvements

- [ ] Add arxiv search untuk research papers
- [ ] Implement memory untuk multi-turn research
- [ ] Add report export (PDF, Markdown)
- [ ] Rate limiting untuk API calls
- [ ] Caching untuk repeated searches

---

**Selanjutnya:** [Proyek 3: Customer Support](/docs/projects/customer-support)
