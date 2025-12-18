---
sidebar_position: 2
title: Tool Calling
description: Cara LLM memilih dan memanggil tools
---

# Tool Calling / Function Calling

Tool Calling adalah kemampuan LLM untuk **memilih** dan **memanggil** functions/tools yang tersedia. Ini adalah fondasi dari agents.

## Bagaimana Tool Calling Bekerja?

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Calling Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. User: "Berapa cuaca di Jakarta?"                       │
│          │                                                   │
│          ▼                                                   │
│   2. LLM melihat available tools:                           │
│      - get_weather(city: str)                               │
│      - search(query: str)                                   │
│      - calculator(expr: str)                                │
│          │                                                   │
│          ▼                                                   │
│   3. LLM memilih: get_weather(city="Jakarta")               │
│          │                                                   │
│          ▼                                                   │
│   4. System menjalankan function                            │
│          │                                                   │
│          ▼                                                   │
│   5. Result dikembalikan ke LLM                             │
│          │                                                   │
│          ▼                                                   │
│   6. LLM generates final answer                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Defining Tools

### Using @tool Decorator

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The city name to get weather for.
    """
    # Simulate API call
    return f"Weather in {city}: 28°C, partly cloudy"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression like '2 + 2' or '10 * 5'.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Check tool details
print(get_weather.name)  # "get_weather"
print(get_weather.description)  # From docstring
print(get_weather.args)  # {'city': {'title': 'City', 'type': 'string'}}
```

**Penting:** Docstring menjadi **description yang dilihat LLM**!

## Binding Tools to LLM

### bind_tools() Method

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather, calculate]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Now LLM knows about tools
response = llm_with_tools.invoke("What's the weather in Jakarta?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Jakarta'}, 'id': 'call_abc123'}]
```

### Understanding Tool Calls Response

```python
response = llm_with_tools.invoke("What is 25 * 4?")

# Response has tool_calls attribute
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['name']}")
        print(f"Args: {call['args']}")
        print(f"ID: {call['id']}")
```

## Executing Tool Calls

LLM hanya **memilih** tool - kita harus **menjalankannya**.

```python
from langchain_core.messages import HumanMessage, ToolMessage

# Step 1: Get tool call from LLM
response = llm_with_tools.invoke("What's the weather in Bandung?")

# Step 2: Execute the tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    
    # Find and execute tool
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call['args'])
    
    # Step 3: Send result back to LLM
    messages = [
        HumanMessage(content="What's the weather in Bandung?"),
        response,  # AI message with tool call
        ToolMessage(
            content=result,
            tool_call_id=tool_call['id']
        )
    ]
    
    # Step 4: Get final answer
    final_response = llm_with_tools.invoke(messages)
    print(final_response.content)
    # "The weather in Bandung is 28°C and partly cloudy."
```

## Automatic Tool Execution

Dengan helper function untuk otomatis execute:

```python
def execute_tool_calls(llm_with_tools, tools, query):
    """Execute tools and get final response."""
    
    # Create tool lookup
    tool_map = {t.name: t for t in tools}
    
    messages = [HumanMessage(content=query)]
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    # Keep executing until no more tool calls
    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool = tool_map[tool_call['name']]
            result = tool.invoke(tool_call['args'])
            
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            ))
        
        response = llm_with_tools.invoke(messages)
        messages.append(response)
    
    return response.content

# Usage
answer = execute_tool_calls(
    llm_with_tools, tools,
    "What's 25 * 4 and what's the weather in Jakarta?"
)
print(answer)
```

## Provider-Specific Implementations

### OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
```

### Anthropic

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)
```

### Google

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_with_tools = llm.bind_tools(tools)
```

### Ollama (Local)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1")
llm_with_tools = llm.bind_tools(tools)
```

## Tool Choice Control

### Let LLM Decide (Default)

```python
llm_with_tools = llm.bind_tools(tools)  # LLM chooses when to use
```

### Force Tool Use

```python
# Force specific tool
llm_with_tools = llm.bind_tools(
    tools, 
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)

# Force ANY tool (must use one)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")

# Disable tools for this call
llm_with_tools = llm.bind_tools(tools, tool_choice="none")
```

## Parallel Tool Calls

LLM dapat memanggil **multiple tools sekaligus**:

```python
response = llm_with_tools.invoke(
    "What's the weather in Jakarta, Bandung, and calculate 10 + 20?"
)

print(len(response.tool_calls))  # 3

for call in response.tool_calls:
    print(f"{call['name']}: {call['args']}")
# get_weather: {'city': 'Jakarta'}
# get_weather: {'city': 'Bandung'}
# calculate: {'expression': '10 + 20'}
```

## Pydantic Tools

Define tools dengan Pydantic untuk validation:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class WeatherInput(BaseModel):
    """Input for weather lookup."""
    city: str = Field(description="City name")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")

def get_weather_impl(city: str, unit: str = "celsius") -> str:
    temp = 28 if unit == "celsius" else 82
    return f"Weather in {city}: {temp}°{'C' if unit == 'celsius' else 'F'}"

weather_tool = StructuredTool.from_function(
    func=get_weather_impl,
    name="get_weather",
    description="Get weather for a city",
    args_schema=WeatherInput
)
```

## Error Handling

```python
@tool
def risky_tool(data: str) -> str:
    """A tool that might fail."""
    try:
        # Some operation
        return process(data)
    except Exception as e:
        return f"Error: {str(e)}"  # Return error as string
```

## Best Practices

### 1. Clear Descriptions

```python
@tool
def search_products(query: str, category: str = None, max_price: float = None) -> str:
    """Search for products in the catalog.
    
    Use this tool when the user wants to find products to buy.
    
    Args:
        query: Search keywords (e.g., 'laptop', 'headphones')
        category: Optional category filter (e.g., 'electronics', 'clothing')
        max_price: Optional maximum price in USD
    
    Returns:
        A list of matching products with prices.
    """
    pass
```

### 2. Return Useful Information

```python
@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    price = fetch_price(symbol)
    
    # Return structured, useful info
    return f"Stock {symbol}: ${price:.2f} (as of {datetime.now()})"
```

### 3. Handle Edge Cases

```python
@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return str(a / b)
```

## Ringkasan

1. **Tool Calling** = LLM memilih functions untuk dipanggil
2. **@tool decorator** - cara mudah define tools
3. **bind_tools()** - attach tools ke LLM
4. **Docstring** = description untuk LLM
5. **Execute manually** atau pakai AgentExecutor
6. Works dengan **semua major providers**

---

**Selanjutnya:** [Built-in Tools](/docs/agents/built-in-tools) - Tools siap pakai dari LangChain.
