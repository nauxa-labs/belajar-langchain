---
sidebar_position: 4
title: "Proyek 3: Customer Support"
description: Sistem customer support dengan classification dan routing
---

# Proyek 3: Customer Support Automation

Membangun sistem customer support dengan **intent classification**, **routing**, dan **human handoff**.

## Requirements

### Fitur Utama
- ✅ Intent classification
- ✅ Route ke departemen yang tepat
- ✅ Auto-response untuk FAQ
- ✅ Human handoff untuk kasus kompleks
- ✅ Ticket creation via tools

### Tech Stack
- LangChain + Structured Output
- Classification + Routing
- Tools untuk actions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Customer Support System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Customer Message                                           │
│        │                                                     │
│        ▼                                                     │
│   [Intent Classifier]                                        │
│        │                                                     │
│   ┌────┴────────────┬────────────────┬───────────┐          │
│   ▼                 ▼                ▼           ▼          │
│ FAQ              Technical        Billing     Complex       │
│   │                 │                │           │          │
│   ▼                 ▼                ▼           ▼          │
│ [Auto-Reply]   [Tech Agent]    [Billing Agent] [Human]     │
│                     │                │                      │
│                     └────────────────┴──────────────────────│
│                              │                              │
│                              ▼                              │
│                      [Create Ticket]                        │
│                              │                              │
│                              ▼                              │
│                     Response + Ticket ID                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Intent Classification

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class Intent(str, Enum):
    FAQ = "faq"
    TECHNICAL = "technical"
    BILLING = "billing"
    COMPLAINT = "complaint"
    HUMAN_NEEDED = "human_needed"

class Classification(BaseModel):
    intent: Intent = Field(description="Customer intent category")
    confidence: float = Field(description="Confidence score 0-1")
    summary: str = Field(description="Brief summary of the issue")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify customer support messages.

Categories:
- faq: General questions about products/services
- technical: Technical issues, bugs, how-to questions
- billing: Payment, invoices, refunds, subscriptions
- complaint: Unhappy customer, escalation needed
- human_needed: Complex issues requiring human agent

Be accurate. Rate confidence 0-1."""),
    ("human", "{message}")
])

classifier = classifier_prompt | llm.with_structured_output(Classification)
```

### Step 2: Department Agents

```python
@tool
def search_faq(query: str) -> str:
    """Search FAQ database for answers."""
    faqs = {
        "hours": "We're open 24/7 for online support.",
        "shipping": "Free shipping on orders over $50.",
        "returns": "30-day return policy on all items.",
    }
    
    for key, answer in faqs.items():
        if key in query.lower():
            return answer
    return "FAQ not found. Routing to human agent."

@tool
def check_order_status(order_id: str) -> str:
    """Check status of an order."""
    # Simulated
    return f"Order {order_id}: Shipped. Arriving in 2 days."

@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request."""
    return f"Refund initiated for order {order_id}. Reason: {reason}. Processing 3-5 business days."

@tool
def create_ticket(
    customer_id: str,
    category: str,
    priority: str,
    description: str
) -> str:
    """Create a support ticket for follow-up."""
    import uuid
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    return f"Ticket created: {ticket_id}. Priority: {priority}. A human agent will follow up."

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to human agent immediately."""
    return f"Escalating to human agent. Reason: {reason}. Please wait..."
```

### Step 3: Routing Logic

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class SupportState(TypedDict):
    message: str
    customer_id: str
    intent: str
    confidence: float
    response: str
    ticket_id: str
    needs_human: bool

def classify_intent(state: SupportState) -> dict:
    result = classifier.invoke({"message": state["message"]})
    return {
        "intent": result.intent.value,
        "confidence": result.confidence
    }

def route_by_intent(state: SupportState) -> str:
    intent = state["intent"]
    confidence = state["confidence"]
    
    # Low confidence = human
    if confidence < 0.7:
        return "human"
    
    if intent == "faq":
        return "faq_handler"
    elif intent == "technical":
        return "tech_handler"
    elif intent == "billing":
        return "billing_handler"
    elif intent in ["complaint", "human_needed"]:
        return "human"
    
    return "human"

def faq_handler(state: SupportState) -> dict:
    result = search_faq.invoke(state["message"])
    return {"response": result}

def tech_handler(state: SupportState) -> dict:
    response = llm.invoke(f"""
You are a technical support agent. Help with:
{state['message']}

Be concise and helpful. If you can't resolve, say so.
""")
    return {"response": response.content}

def billing_handler(state: SupportState) -> dict:
    # Check for order-related queries
    if "order" in state["message"].lower():
        # Extract order ID (simplified)
        response = check_order_status.invoke("ORD-12345")
    elif "refund" in state["message"].lower():
        response = process_refund.invoke({"order_id": "ORD-12345", "reason": "Customer request"})
    else:
        response = llm.invoke(f"Answer billing question: {state['message']}").content
    
    return {"response": response}

def human_handler(state: SupportState) -> dict:
    ticket = create_ticket.invoke({
        "customer_id": state["customer_id"],
        "category": state["intent"],
        "priority": "high",
        "description": state["message"]
    })
    
    return {
        "response": f"I understand this needs special attention. {ticket}",
        "needs_human": True,
        "ticket_id": ticket.split(":")[1].strip().split(".")[0]
    }
```

### Step 4: Build Graph

```python
builder = StateGraph(SupportState)

builder.add_node("classify", classify_intent)
builder.add_node("faq_handler", faq_handler)
builder.add_node("tech_handler", tech_handler)
builder.add_node("billing_handler", billing_handler)
builder.add_node("human", human_handler)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_intent)
builder.add_edge("faq_handler", END)
builder.add_edge("tech_handler", END)
builder.add_edge("billing_handler", END)
builder.add_edge("human", END)

support_graph = builder.compile()
```

## Complete Code

```python
#!/usr/bin/env python3
"""
Customer Support Automation
Proyek 3 - Modul 10
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
import uuid

load_dotenv()


# ===== CLASSIFICATION =====

class Classification(BaseModel):
    intent: Literal["faq", "technical", "billing", "complaint", "human_needed"]
    confidence: float
    summary: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify customer support messages:
- faq: General questions
- technical: Tech issues
- billing: Payment/refund
- complaint: Unhappy customer
- human_needed: Complex/sensitive"""),
    ("human", "{message}")
])

classifier = classifier_prompt | llm.with_structured_output(Classification)


# ===== TOOLS =====

@tool
def search_faq(query: str) -> str:
    """Search FAQ database."""
    faqs = {
        "shipping": "Free shipping over $50. Standard 3-5 days.",
        "return": "30-day returns on all items.",
        "hours": "24/7 online support available.",
        "contact": "Email: support@example.com, Phone: 1-800-XXX"
    }
    for key, val in faqs.items():
        if key in query.lower():
            return val
    return "No FAQ match found."

@tool
def create_ticket(customer_id: str, priority: str, issue: str) -> str:
    """Create support ticket."""
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    return f"Ticket {ticket_id} created. Priority: {priority}"


# ===== STATE & HANDLERS =====

class SupportState(TypedDict):
    message: str
    customer_id: str
    intent: str
    confidence: float
    response: str
    ticket_id: str

def classify_node(state: SupportState) -> dict:
    result = classifier.invoke({"message": state["message"]})
    return {"intent": result.intent, "confidence": result.confidence}

def route(state: SupportState) -> str:
    if state["confidence"] < 0.6:
        return "human_handler"
    return f"{state['intent']}_handler"

def faq_handler(state: SupportState) -> dict:
    result = search_faq.invoke(state["message"])
    return {"response": result}

def technical_handler(state: SupportState) -> dict:
    response = llm.invoke(f"Technical support: {state['message']}")
    return {"response": response.content}

def billing_handler(state: SupportState) -> dict:
    response = llm.invoke(f"Billing support: {state['message']}")
    return {"response": response.content}

def complaint_handler(state: SupportState) -> dict:
    ticket = create_ticket.invoke({
        "customer_id": state["customer_id"],
        "priority": "high",
        "issue": state["message"]
    })
    return {"response": f"I apologize for the inconvenience. {ticket}", "ticket_id": ticket}

def human_needed_handler(state: SupportState) -> dict:
    ticket = create_ticket.invoke({
        "customer_id": state["customer_id"],
        "priority": "urgent",
        "issue": state["message"]
    })
    return {"response": f"Connecting you to a specialist. {ticket}", "ticket_id": ticket}

def human_handler(state: SupportState) -> dict:
    return human_needed_handler(state)


# ===== GRAPH =====

builder = StateGraph(SupportState)

builder.add_node("classify", classify_node)
builder.add_node("faq_handler", faq_handler)
builder.add_node("technical_handler", technical_handler)
builder.add_node("billing_handler", billing_handler)
builder.add_node("complaint_handler", complaint_handler)
builder.add_node("human_needed_handler", human_needed_handler)
builder.add_node("human_handler", human_handler)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)
builder.add_edge("faq_handler", END)
builder.add_edge("technical_handler", END)
builder.add_edge("billing_handler", END)
builder.add_edge("complaint_handler", END)
builder.add_edge("human_needed_handler", END)
builder.add_edge("human_handler", END)

support_system = builder.compile()


# ===== MAIN =====

def handle_customer(message: str, customer_id: str = "CUST-001") -> dict:
    result = support_system.invoke({
        "message": message,
        "customer_id": customer_id,
        "intent": "",
        "confidence": 0.0,
        "response": "",
        "ticket_id": ""
    })
    return result

def main():
    print("🎧 Customer Support System")
    print("Type 'quit' to exit\n")
    
    while True:
        message = input("Customer: ").strip()
        
        if message.lower() in ('quit', 'exit'):
            break
        
        if not message:
            continue
        
        result = handle_customer(message)
        
        print(f"\n📋 Intent: {result['intent']} ({result['confidence']:.0%})")
        print(f"🤖 Response: {result['response']}")
        if result.get('ticket_id'):
            print(f"🎫 Ticket: {result['ticket_id']}")
        print()

if __name__ == "__main__":
    main()
```

## Testing

```bash
# Test cases
Customer: What are your shipping rates?
# → FAQ: Free shipping over $50...

Customer: My order hasn't arrived
# → Technical/Billing: Creates ticket

Customer: I want to speak to a manager!
# → Complaint: High priority ticket

Customer: 这是什么 (non-English)
# → Human needed: Low confidence routing
```

## Improvements

- [ ] Add sentiment analysis
- [ ] Implement queue system
- [ ] Add analytics dashboard
- [ ] Multi-language support
- [ ] Knowledge base integration

---

**Selanjutnya:** [Proyek 4: Content Pipeline](/docs/projects/content-pipeline)
