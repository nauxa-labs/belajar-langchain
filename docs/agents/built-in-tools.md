---
sidebar_position: 3
title: Built-in Tools
description: Tools siap pakai dari LangChain
---

# Built-in Tools

LangChain menyediakan banyak **tools siap pakai** untuk berbagai keperluan.

## Search Tools

### Tavily Search (Recommended)

AI-optimized search engine.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Requires TAVILY_API_KEY
search = TavilySearchResults(max_results=3)

results = search.invoke("Latest news about AI")
print(results)
```

```bash
pip install tavily-python
```

### DuckDuckGo Search (Free)

No API key required.

```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

result = search.invoke("Python programming tutorials")
print(result)
```

```bash
pip install duckduckgo-search
```

### Google Search

```python
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

search = GoogleSearchAPIWrapper()

google_tool = Tool(
    name="google_search",
    description="Search Google for recent results",
    func=search.run
)
```

Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`.

## Wikipedia

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

result = wikipedia.invoke("Machine Learning")
print(result[:500])
```

```bash
pip install wikipedia
```

## ArXiv (Research Papers)

```python
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

result = arxiv.invoke("transformer neural network")
print(result)
```

```bash
pip install arxiv
```

## Calculator / Math

### LLMMathChain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
from langchain_core.tools import Tool

llm = ChatOpenAI(model="gpt-4o-mini")
llm_math = LLMMathChain.from_llm(llm)

math_tool = Tool(
    name="calculator",
    description="Useful for math calculations. Input should be a math expression.",
    func=llm_math.run
)

result = math_tool.invoke("What is 15% of 250?")
print(result)  # "37.5"
```

### Simple Eval Tool

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expression. 
    
    Examples: '2 + 2', '10 * 5', 'sqrt(16)', 'sin(3.14/2)'
    """
    import math
    
    # Safe eval with math functions
    allowed = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

## Python REPL

Execute Python code (⚠️ Use with caution!)

```python
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()

# Execute Python code
result = python_repl.invoke("""
import math
radius = 5
area = math.pi * radius ** 2
print(f"Area: {area:.2f}")
""")
print(result)
```

```bash
pip install langchain-experimental
```

**⚠️ Security Warning:** This executes arbitrary Python code. Only use in sandboxed environments!

## File System Tools

### Read Files

```python
from langchain_community.tools.file_management import ReadFileTool

read_tool = ReadFileTool()

content = read_tool.invoke({"file_path": "readme.txt"})
print(content)
```

### Write Files

```python
from langchain_community.tools.file_management import WriteFileTool

write_tool = WriteFileTool()

write_tool.invoke({
    "file_path": "output.txt",
    "text": "Hello, World!"
})
```

### List Directory

```python
from langchain_community.tools.file_management import ListDirectoryTool

list_tool = ListDirectoryTool()

files = list_tool.invoke({"dir_path": "."})
print(files)
```

### File Management Toolkit

```python
from langchain_community.agent_toolkits import FileManagementToolkit

toolkit = FileManagementToolkit(
    root_dir="./workspace",
    selected_tools=["read_file", "write_file", "list_directory"]
)

tools = toolkit.get_tools()
```

## Web Requests

```python
from langchain_community.tools import RequestsGetTool, RequestsPostTool
from langchain_community.utilities import TextRequestsWrapper

requests = TextRequestsWrapper()

get_tool = RequestsGetTool(requests_wrapper=requests)
post_tool = RequestsPostTool(requests_wrapper=requests)

# GET request
response = get_tool.invoke("https://api.github.com/users/octocat")
print(response)
```

## Shell Commands

```python
from langchain_community.tools import ShellTool

shell = ShellTool()

result = shell.invoke("ls -la")
print(result)
```

**⚠️ Security Warning:** Don't use in production without strict input validation!

## Date/Time

```python
from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_day_of_week(date_str: str) -> str:
    """Get the day of week for a date (format: YYYY-MM-DD)."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return date.strftime("%A")
```

## Human Input

Let agent ask user for input:

```python
from langchain_community.tools import HumanInputRun

human_tool = HumanInputRun()

# Agent can use this to ask user questions
# response = human_tool.invoke("What is your preferred language?")
```

## SQL Database

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDataBaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool
)

db = SQLDatabase.from_uri("sqlite:///example.db")

# List tables
list_tool = ListSQLDatabaseTool(db=db)
print(list_tool.invoke(""))

# Get table info
info_tool = InfoSQLDatabaseTool(db=db)
print(info_tool.invoke("users"))

# Run query
query_tool = QuerySQLDataBaseTool(db=db)
result = query_tool.invoke("SELECT * FROM users LIMIT 5")
```

## Creating Toolkits

Group related tools:

```python
from langchain_core.tools import tool

@tool
def create_user(name: str, email: str) -> str:
    """Create a new user."""
    return f"Created user: {name} ({email})"

@tool
def get_user(user_id: str) -> str:
    """Get user by ID."""
    return f"User {user_id}: John Doe (john@example.com)"

@tool
def delete_user(user_id: str) -> str:
    """Delete a user."""
    return f"Deleted user: {user_id}"

user_management_tools = [create_user, get_user, delete_user]
```

## Complete Agent Example with Built-in Tools

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

# Tools
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

tools = [search, wikipedia, calculator]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant.
You have access to:
- Search: for current information
- Wikipedia: for factual knowledge
- Calculator: for math

Always verify information from multiple sources when possible."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = executor.invoke({
    "input": "What is the current population of Indonesia and what's 10% of that?"
})
print(result["output"])
```

## Summary Table

| Tool | Purpose | API Key Required |
|------|---------|------------------|
| TavilySearchResults | Web search (AI-optimized) | Yes |
| DuckDuckGoSearchRun | Web search (free) | No |
| WikipediaQueryRun | Encyclopedia lookup | No |
| ArxivQueryRun | Research papers | No |
| PythonREPLTool | Execute Python | No |
| FileManagementToolkit | File operations | No |
| ShellTool | Shell commands | No |
| SQLDatabase tools | Database queries | No |

## Ringkasan

1. **Search**: Tavily (paid, best), DuckDuckGo (free)
2. **Knowledge**: Wikipedia, ArXiv
3. **Compute**: Calculator, Python REPL
4. **Files**: Read, Write, List directory
5. **Database**: SQL query tools
6. Combine tools into **toolkits** untuk use cases tertentu

---

**Selanjutnya:** [Custom Tools](/docs/agents/custom-tools) - Membuat tools sendiri.
