---
sidebar_position: 4
title: Custom Tools
description: Membuat tools sendiri untuk agents
---

# Custom Tools

Seringkali built-in tools tidak cukup - kita perlu membuat **tools khusus** untuk aplikasi kita.

## @tool Decorator (Simplest)

Cara paling mudah membuat tool.

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The product of a and b
    """
    return a * b

# Check tool attributes
print(multiply.name)           # "multiply"
print(multiply.description)    # From docstring
print(multiply.args_schema)    # Pydantic schema
```

**Key Points:**
- **Function name** = tool name
- **Docstring** = description (LLM sees this!)
- **Type hints** = argument schema
- **Return type** = output type

## Pydantic Schema (Validation)

Untuk complex inputs dengan validation:

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

class SearchInput(BaseModel):
    """Input for product search."""
    query: str = Field(description="Search keywords")
    category: Optional[str] = Field(
        default=None, 
        description="Filter by category: electronics, clothing, books"
    )
    max_price: Optional[float] = Field(
        default=None, 
        description="Maximum price in USD"
    )
    in_stock: bool = Field(
        default=True, 
        description="Only show items in stock"
    )

@tool(args_schema=SearchInput)
def search_products(
    query: str, 
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock: bool = True
) -> str:
    """Search for products in the catalog.
    
    Use this when user wants to find products to buy.
    Returns a list of matching products with prices.
    """
    # Implementation
    results = f"Found products for '{query}'"
    if category:
        results += f" in {category}"
    if max_price:
        results += f" under ${max_price}"
    return results
```

## StructuredTool Class

Untuk lebih kontrol:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name")
    unit: str = Field(default="celsius", description="celsius or fahrenheit")

def get_weather_impl(city: str, unit: str = "celsius") -> str:
    """Implementation of weather lookup."""
    temp = 28 if unit == "celsius" else 82
    symbol = "°C" if unit == "celsius" else "°F"
    return f"Weather in {city}: {temp}{symbol}, sunny"

weather_tool = StructuredTool.from_function(
    func=get_weather_impl,
    name="get_weather",
    description="Get current weather for a city. Use when user asks about weather.",
    args_schema=WeatherInput,
    return_direct=False  # If True, returns tool output directly
)
```

## Async Tools

Untuk I/O operations:

```python
from langchain_core.tools import tool
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL.
    
    Args:
        url: The URL to fetch
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()
            return content[:1000]  # First 1000 chars
```

```bash
pip install aiohttp
```

## Tools with Context

Access conversation context:

```python
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

@tool
def get_user_info(config: RunnableConfig) -> str:
    """Get information about current user."""
    user_id = config.get("configurable", {}).get("user_id")
    if user_id:
        return f"User ID: {user_id}"
    return "No user context available"
```

## API Integration Tools

### REST API Tool

```python
from langchain_core.tools import tool
import requests

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL)
    """
    # In real app, use actual API
    # response = requests.get(f"https://api.example.com/stock/{symbol}")
    
    # Simulated response
    prices = {"AAPL": 175.50, "GOOGL": 140.25, "MSFT": 380.00}
    if symbol.upper() in prices:
        return f"{symbol.upper()}: ${prices[symbol.upper()]:.2f}"
    return f"Symbol {symbol} not found"

@tool
def get_crypto_price(coin: str) -> str:
    """Get current cryptocurrency price.
    
    Args:
        coin: Cryptocurrency name (e.g., bitcoin, ethereum)
    """
    prices = {"bitcoin": 43500, "ethereum": 2250, "solana": 95}
    coin_lower = coin.lower()
    if coin_lower in prices:
        return f"{coin.title()}: ${prices[coin_lower]:,}"
    return f"Coin {coin} not found"
```

### Database Tool

```python
from langchain_core.tools import tool
import sqlite3

@tool
def query_products(category: str, limit: int = 5) -> str:
    """Query products from database.
    
    Args:
        category: Product category to filter
        limit: Maximum number of results (default 5)
    """
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT name, price FROM products WHERE category = ? LIMIT ?",
            (category, limit)
        )
        results = cursor.fetchall()
        
        if not results:
            return f"No products found in category '{category}'"
        
        output = f"Products in {category}:\n"
        for name, price in results:
            output += f"- {name}: ${price:.2f}\n"
        return output
    
    finally:
        conn.close()
```

## Tool with State

```python
from langchain_core.tools import tool
from typing import Dict

# Shared state
cart: Dict[str, int] = {}

@tool
def add_to_cart(product: str, quantity: int = 1) -> str:
    """Add a product to the shopping cart.
    
    Args:
        product: Product name to add
        quantity: Number of items (default 1)
    """
    if product in cart:
        cart[product] += quantity
    else:
        cart[product] = quantity
    
    return f"Added {quantity}x {product} to cart. Cart now has {sum(cart.values())} items."

@tool
def view_cart() -> str:
    """View current shopping cart contents."""
    if not cart:
        return "Cart is empty"
    
    output = "Shopping Cart:\n"
    for product, qty in cart.items():
        output += f"- {product}: {qty}\n"
    output += f"\nTotal items: {sum(cart.values())}"
    return output

@tool
def clear_cart() -> str:
    """Clear all items from cart."""
    cart.clear()
    return "Cart cleared"
```

## Error Handling

### Return Errors as Strings

```python
@tool
def safe_divide(a: float, b: float) -> str:
    """Divide two numbers safely.
    
    Args:
        a: Numerator
        b: Denominator
    """
    if b == 0:
        return "Error: Cannot divide by zero"
    
    try:
        result = a / b
        return f"{a} / {b} = {result:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### ToolException

```python
from langchain_core.tools import tool, ToolException

@tool
def strict_divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
    """
    if b == 0:
        raise ToolException("Cannot divide by zero")
    return a / b

# In agent executor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  # Handles ToolException gracefully
)
```

## Tool Validation

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

class EmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    
    @field_validator('to')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @field_validator('subject')
    @classmethod
    def validate_subject(cls, v):
        if len(v) > 100:
            raise ValueError('Subject too long (max 100 chars)')
        return v

@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email message.
    
    Use this to send emails to users.
    """
    # Implementation
    return f"Email sent to {to}"
```

## Complete Example: E-commerce Agent Tools

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Simulated product database
PRODUCTS = {
    "laptop": {"name": "Pro Laptop", "price": 1299.99, "stock": 10},
    "phone": {"name": "Smart Phone", "price": 699.99, "stock": 25},
    "headphones": {"name": "Wireless Headphones", "price": 149.99, "stock": 50},
}

@tool
def search_products(query: str, max_price: Optional[float] = None) -> str:
    """Search for products in the store.
    
    Args:
        query: Search keywords
        max_price: Maximum price filter (optional)
    """
    results = []
    for key, product in PRODUCTS.items():
        if query.lower() in key.lower() or query.lower() in product["name"].lower():
            if max_price is None or product["price"] <= max_price:
                results.append(
                    f"- {product['name']}: ${product['price']:.2f} ({product['stock']} in stock)"
                )
    
    if not results:
        return f"No products found for '{query}'"
    
    return "Found products:\n" + "\n".join(results)

@tool
def get_product_details(product_id: str) -> str:
    """Get detailed information about a product.
    
    Args:
        product_id: Product identifier (e.g., 'laptop', 'phone')
    """
    if product_id.lower() not in PRODUCTS:
        return f"Product '{product_id}' not found"
    
    p = PRODUCTS[product_id.lower()]
    return f"""
Product: {p['name']}
Price: ${p['price']:.2f}
In Stock: {p['stock']} units
"""

@tool
def check_stock(product_id: str) -> str:
    """Check if a product is in stock.
    
    Args:
        product_id: Product identifier
    """
    if product_id.lower() not in PRODUCTS:
        return f"Product '{product_id}' not found"
    
    stock = PRODUCTS[product_id.lower()]["stock"]
    if stock > 0:
        return f"{PRODUCTS[product_id.lower()]['name']}: {stock} units available"
    return f"{PRODUCTS[product_id.lower()]['name']}: Out of stock"

@tool
def place_order(product_id: str, quantity: int = 1) -> str:
    """Place an order for a product.
    
    Args:
        product_id: Product to order
        quantity: Number of items (default 1)
    """
    if product_id.lower() not in PRODUCTS:
        return f"Product '{product_id}' not found"
    
    p = PRODUCTS[product_id.lower()]
    if p["stock"] < quantity:
        return f"Sorry, only {p['stock']} units available"
    
    total = p["price"] * quantity
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Update stock (in real app, this would be a DB transaction)
    PRODUCTS[product_id.lower()]["stock"] -= quantity
    
    return f"""
✅ Order Placed!
Order ID: {order_id}
Product: {p['name']}
Quantity: {quantity}
Total: ${total:.2f}
"""

# Create toolkit
ecommerce_tools = [search_products, get_product_details, check_stock, place_order]
```

## Best Practices

### 1. Descriptive Docstrings

```python
@tool
def book_flight(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1
) -> str:
    """Book a flight between two cities.
    
    Use this tool when the user wants to book air travel.
    
    Args:
        origin: Departure city or airport code (e.g., 'Jakarta' or 'CGK')
        destination: Arrival city or airport code
        date: Travel date in YYYY-MM-DD format
        passengers: Number of passengers (default 1)
    
    Returns:
        Booking confirmation with flight details and price.
    """
    pass
```

### 2. Meaningful Return Values

```python
# ❌ Bad - unhelpful
return "Done"

# ✅ Good - informative
return f"Created user '{username}' with ID {user_id}. Email sent to {email}."
```

### 3. Handle Edge Cases

```python
@tool
def get_user(user_id: str) -> str:
    """Get user information."""
    if not user_id:
        return "Error: user_id is required"
    
    user = db.get_user(user_id)
    if not user:
        return f"No user found with ID '{user_id}'"
    
    return f"User: {user.name} ({user.email})"
```

## Ringkasan

1. **@tool decorator** - quickest way
2. **Pydantic schema** - for validation
3. **StructuredTool** - full control
4. **Docstrings are critical** - LLM reads them
5. **Return useful messages** - not just "done"
6. **Handle errors gracefully**

---

**Selanjutnya:** [Agent Executors](/docs/agents/agent-executors) - Menjalankan agents dengan loop.
