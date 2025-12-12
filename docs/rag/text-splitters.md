---
sidebar_position: 3
title: Text Splitters
description: Memecah dokumen menjadi chunks optimal untuk RAG
---

# Text Splitters

Setelah dokumen di-load, langkah selanjutnya adalah **memecahnya menjadi chunks yang lebih kecil**. Ini penting karena:

1. LLM memiliki **context window terbatas**
2. Chunks kecil = **retrieval lebih precise**
3. Embedding bekerja lebih baik pada **teks yang fokus**

## Konsep Dasar

```
┌──────────────────────────────────────────────────┐
│                  Full Document                    │
│  "Lorem ipsum dolor sit amet, consectetur        │
│   adipiscing elit. Sed do eiusmod tempor         │
│   incididunt ut labore et dolore magna aliqua.   │
│   Ut enim ad minim veniam..."                    │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    Chunk 1    │ │    Chunk 2    │ │    Chunk 3    │
│  "Lorem ipsum │ │ "...tempor    │ │ "...Ut enim   │
│   dolor sit   │ │  incididunt   │ │   ad minim    │
│   amet..."    │ │  ut labore..."│ │   veniam..."  │
└───────────────┘ └───────────────┘ └───────────────┘
        ▲─────────overlap──────────▲
```

## Parameter Penting

### chunk_size

Ukuran maksimum setiap chunk (dalam characters atau tokens).

```python
chunk_size=1000  # ~250 tokens, ~1 paragraf
```

**Guidelines:**
- 500-1000: Untuk Q&A yang precise
- 1000-2000: Untuk summarization
- 2000+: Untuk dokumen yang butuh context besar

### chunk_overlap

Seberapa banyak teks yang di-share antar chunks.

```python
chunk_overlap=200  # 20% overlap is common
```

**Mengapa perlu overlap?**
- Mencegah informasi penting terpotong
- Menjaga konteks antar chunks
- Biasanya 10-20% dari chunk_size

## RecursiveCharacterTextSplitter

**Paling sering digunakan** - splits berdasarkan hierarchy of separators.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Try in order
)

text = """
Bab 1: Pengenalan

Python adalah bahasa pemrograman yang populer.
Python mudah dipelajari untuk pemula.

Bab 2: Instalasi

Download Python dari python.org.
Ikuti instruksi instalasi.
"""

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars")
    print(chunk[:100] + "...")
    print()
```

### Separator Hierarchy

```python
# Default separators (dari paling preferred ke fallback)
separators=[
    "\n\n",  # Paragraph breaks
    "\n",    # Line breaks
    " ",     # Spaces
    ""       # Character by character (last resort)
]
```

### Custom Separators

```python
# Untuk markdown
markdown_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[
        "\n## ",   # H2 headers
        "\n### ",  # H3 headers
        "\n\n",    # Paragraphs
        "\n",      # Lines
        " ",       # Words
    ]
)
```

## CharacterTextSplitter

Split berdasarkan **satu separator saja**.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",  # Only split on double newlines
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)
```

**Use case:** Dokumen dengan struktur jelas (paragraphs, sections).

## TokenTextSplitter

Split berdasarkan **token count** (lebih akurat untuk LLM).

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,      # 500 tokens
    chunk_overlap=50     # 50 token overlap
)

chunks = splitter.split_text(text)
```

```bash
pip install tiktoken
```

**Kapan pakai:**
- Untuk kontrollebih precise terhadap token usage
- Saat embedding model punya token limit

## Splitting Documents

Untuk split `Document` objects (dengan metadata):

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Metadata preserved!
print(chunks[0].metadata)  # {"source": "document.pdf", "page": 0}
```

## Language-aware Splitters

Untuk code, gunakan splitter yang paham syntax.

### Python

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)

python_code = """
def hello(name):
    '''Say hello.'''
    return f"Hello, {name}!"

class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
"""

chunks = python_splitter.split_text(python_code)
```

### Supported Languages

```python
from langchain_text_splitters import Language

# Check supported languages
print([e.value for e in Language])
# ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 
#  'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 
#  'latex', 'html', 'sol', 'csharp', 'cobol']
```

## MarkdownHeaderTextSplitter

Split markdown berdasarkan headers, preserving hierarchy.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

markdown = """
# Main Title

Introduction paragraph.

## Section 1

Content for section 1.

### Subsection 1.1

Details for subsection.

## Section 2

Content for section 2.
"""

chunks = splitter.split_text(markdown)

for chunk in chunks:
    print(f"Content: {chunk.page_content[:50]}...")
    print(f"Metadata: {chunk.metadata}")
    print()
```

Output:
```
Content: Introduction paragraph....
Metadata: {"Header 1": "Main Title"}

Content: Content for section 1....
Metadata: {"Header 1": "Main Title", "Header 2": "Section 1"}

Content: Details for subsection....
Metadata: {"Header 1": "Main Title", "Header 2": "Section 1", "Header 3": "Subsection 1.1"}
```

## HTMLHeaderTextSplitter

Split HTML berdasarkan heading tags.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on)
chunks = splitter.split_text(html_content)
```

## Semantic Chunking

Split berdasarkan **perubahan semantik** dalam teks.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text_splitter = SemanticChunker(OpenAIEmbeddings())

chunks = text_splitter.split_text(long_text)
```

**Cara kerja:**
1. Embed setiap kalimat
2. Deteksi perubahan similaritas yang signifikan
3. Split di boundary tersebut

## Combining Splitters

Combine header splitting + recursive splitting:

```python
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# Step 1: Split by headers
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")]
)
md_docs = md_splitter.split_text(markdown_content)

# Step 2: Further split large sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_chunks = text_splitter.split_documents(md_docs)
```

## Measuring Chunk Quality

```python
def analyze_chunks(chunks):
    lengths = [len(c.page_content) for c in chunks]
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"Min length: {min(lengths)} chars")
    print(f"Max length: {max(lengths)} chars")
    print(f"Std dev: {(sum((x-sum(lengths)/len(lengths))**2 for x in lengths)/len(lengths))**0.5:.0f}")

analyze_chunks(chunks)
```

## Best Practices

### 1. Start with RecursiveCharacterTextSplitter

```python
# Good default for most use cases
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### 2. Tune Based on Content

| Content Type | chunk_size | chunk_overlap |
|--------------|------------|---------------|
| Q&A | 500-1000 | 100-200 |
| Summarization | 1000-2000 | 200-300 |
| Code | 1500-2500 | 200-400 |
| Legal/Technical | 800-1200 | 150-250 |

### 3. Consider Your Embedding Model

```python
# OpenAI ada-002: 8191 tokens max
# Bge-small: 512 tokens max
# Check your model's limit!
```

### 4. Preserve Context

```python
# Add metadata context to chunks
for chunk in chunks:
    chunk.page_content = f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}"
```

## Ringkasan

1. **RecursiveCharacterTextSplitter** - best default choice
2. **chunk_size** - 500-1000 untuk Q&A, lebih besar untuk summarization
3. **chunk_overlap** - 10-20% untuk preserve context
4. **Language-aware splitters** - untuk code
5. **MarkdownHeaderTextSplitter** - preserve document structure
6. **Combine splitters** untuk hasil optimal

---

**Selanjutnya:** [Embeddings](/docs/rag/embeddings) - Mengubah teks menjadi vectors.
