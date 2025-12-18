---
sidebar_position: 4
title: LangServe
description: Deploy chain sebagai REST API
---

# LangServe

LangServe memudahkan **deploy LangChain sebagai REST API** dengan FastAPI.

## Installation

```bash
pip install langserve[all]
```

## Quick Start

### server.py

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Create chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | ChatOpenAI(model="gpt-4o-mini")

# Create FastAPI app
app = FastAPI(
    title="Joke API",
    version="1.0",
    description="API for generating jokes"
)

# Add routes for the chain
add_routes(app, chain, path="/joke")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run

```bash
python server.py
```

### Access

- API: `http://localhost:8000/joke/invoke`
- Playground: `http://localhost:8000/joke/playground`
- Docs: `http://localhost:8000/docs`

## API Endpoints

LangServe automatically creates:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/invoke` | POST | Run chain once |
| `/batch` | POST | Run on multiple inputs |
| `/stream` | POST | Stream response |
| `/stream_log` | POST | Stream with intermediate steps |
| `/playground` | GET | Interactive testing UI |

### Invoke

```bash
curl -X POST "http://localhost:8000/joke/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"topic": "programming"}}'
```

Response:
```json
{
  "output": "Why do programmers prefer dark mode? Because light attracts bugs!",
  "metadata": {...}
}
```

### Batch

```bash
curl -X POST "http://localhost:8000/joke/batch" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"topic": "cats"}, {"topic": "dogs"}]}'
```

### Stream

```bash
curl -X POST "http://localhost:8000/joke/stream" \
  -H "Content-Type: application/json" \
  -d '{"input": {"topic": "AI"}}'
```

## Multiple Chains

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()

# Chain 1: Jokes
add_routes(app, joke_chain, path="/joke")

# Chain 2: Q&A
add_routes(app, qa_chain, path="/qa")

# Chain 3: Summarization
add_routes(app, summary_chain, path="/summarize")

# All available at their respective paths
```

## Client SDK

### Python Client

```python
from langserve import RemoteRunnable

# Connect to server
joke_chain = RemoteRunnable("http://localhost:8000/joke")

# Use like a regular chain
result = joke_chain.invoke({"topic": "programming"})
print(result)

# Streaming
for chunk in joke_chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Batch
results = joke_chain.batch([
    {"topic": "cats"},
    {"topic": "dogs"}
])
```

### JavaScript Client

```javascript
import { RemoteRunnable } from "@langchain/core/runnables/remote";

const chain = new RemoteRunnable({
  url: "http://localhost:8000/joke",
});

// Invoke
const result = await chain.invoke({ topic: "JavaScript" });
console.log(result);

// Stream
for await (const chunk of await chain.stream({ topic: "coding" })) {
  console.log(chunk);
}
```

## Authentication

### API Key Auth

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
import os

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

app = FastAPI()

# Protected routes
add_routes(
    app, 
    chain, 
    path="/protected",
    dependencies=[Depends(verify_api_key)]
)
```

### OAuth2

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token
    user = verify_token(token)
    return user

add_routes(
    app,
    chain,
    path="/secure",
    dependencies=[Depends(get_current_user)]
)
```

## Input/Output Schemas

### Custom Input Schema

```python
from pydantic import BaseModel, Field
from langserve import add_routes

class JokeInput(BaseModel):
    topic: str = Field(description="Topic for the joke")
    style: str = Field(default="funny", description="Style: funny, sarcastic, dad-joke")

add_routes(
    app,
    chain,
    path="/joke",
    input_type=JokeInput
)
```

### Custom Output Schema

```python
class JokeOutput(BaseModel):
    joke: str
    rating: int = Field(description="1-10 humor rating")

add_routes(
    app,
    chain.with_types(output_type=JokeOutput),
    path="/joke"
)
```

## Middleware & CORS

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/rate-limited-joke")
@limiter.limit("10/minute")
async def limited_joke(request: Request, input: JokeInput):
    return await joke_chain.ainvoke(input.dict())
```

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
```

### Cloud Deployment

```bash
# Render, Railway, Fly.io
# Just connect repo with Dockerfile

# AWS Lambda (with Mangum)
pip install mangum

from mangum import Mangum
handler = Mangum(app)
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Production LangServe API
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Chains
llm = ChatOpenAI(model="gpt-4o-mini")

joke_prompt = ChatPromptTemplate.from_template("Tell a {style} joke about {topic}")
joke_chain = joke_prompt | llm | StrOutputParser()

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

# App
app = FastAPI(
    title="AI API",
    version="1.0.0",
    description="Production AI API with jokes and Q&A"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth
API_KEY = os.getenv("API_KEY", "dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")

# Routes
add_routes(app, joke_chain, path="/joke")
add_routes(app, qa_chain, path="/qa", dependencies=[Depends(verify_key)])

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Ringkasan

1. **add_routes()** - instant REST API for chains
2. **Auto endpoints** - invoke, batch, stream, playground
3. **Client SDK** - Python & JavaScript
4. **Authentication** - API key, OAuth2
5. **Deployment** - Docker, cloud platforms

---

**Selanjutnya:** [Best Practices](/docs/production/best-practices) - Production-ready patterns.
