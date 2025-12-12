---
sidebar_position: 2
title: Document Loaders
description: Membaca dokumen dari berbagai sumber untuk RAG
---

# Document Loaders

Document Loaders adalah komponen yang membaca data dari berbagai sumber dan mengubahnya menjadi format `Document` yang bisa diproses LangChain.

## Document Object

Setiap dokumen di LangChain memiliki struktur:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Isi teks dokumen...",
    metadata={"source": "file.pdf", "page": 1}
)

print(doc.page_content)  # Konten teks
print(doc.metadata)      # Informasi tambahan
```

## Text Files

### TextLoader

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("readme.txt", encoding="utf-8")
docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)  # {"source": "readme.txt"}
```

### DirectoryLoader

Load semua file dalam directory.

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all .txt files
loader = DirectoryLoader(
    "./documents/",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
```

## PDF Files

### PyPDFLoader

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
pages = loader.load()

# Each page is a separate Document
for page in pages:
    print(f"Page {page.metadata['page']}: {len(page.page_content)} chars")
```

### Installation

```bash
pip install pypdf
```

### PyPDFDirectoryLoader

Load semua PDF dari folder.

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("./pdfs/")
docs = loader.load()
```

### PDFPlumberLoader

Lebih baik untuk PDF dengan tables.

```python
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("table_document.pdf")
docs = loader.load()
```

```bash
pip install pdfplumber
```

## Web Pages

### WebBaseLoader

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://python.langchain.com/docs/")
docs = loader.load()

print(docs[0].page_content[:500])
```

### Multiple URLs

```python
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

loader = WebBaseLoader(urls)
docs = loader.load()
```

### RecursiveUrlLoader

Crawl website secara rekursif.

```python
from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    url="https://docs.example.com/",
    max_depth=2  # How deep to crawl
)
docs = loader.load()
```

## Markdown Files

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md")
docs = loader.load()
```

Atau dengan header splitting:

```python
from langchain_community.document_loaders import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
splits = splitter.split_text(markdown_content)
```

## CSV Files

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"'
    }
)
docs = loader.load()

# Each row becomes a Document
for doc in docs[:3]:
    print(doc.page_content)
```

## JSON Files

```python
from langchain_community.document_loaders import JSONLoader

# Simple JSON
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[]",  # jq query to extract content
    text_content=False
)
docs = loader.load()
```

```bash
pip install jq
```

## Word Documents

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
docs = loader.load()
```

```bash
pip install docx2txt
```

## HTML Files

```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("page.html")
docs = loader.load()
```

Atau dengan BeautifulSoup:

```python
from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("page.html")
docs = loader.load()
```

## Notion

```python
from langchain_community.document_loaders import NotionDirectoryLoader

# Export Notion workspace first, then load
loader = NotionDirectoryLoader("./notion_export/")
docs = loader.load()
```

## YouTube Transcripts

```python
from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    add_video_info=True
)
docs = loader.load()

print(docs[0].metadata)  # {"title": "...", "author": "..."}
```

```bash
pip install youtube-transcript-api pytube
```

## GitHub

```python
from langchain_community.document_loaders import GitHubIssuesLoader

loader = GitHubIssuesLoader(
    repo="langchain-ai/langchain",
    access_token="ghp_...",
    include_prs=False
)
issues = loader.load()
```

## Custom Loader

Buat loader sendiri untuk format khusus.

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import List

class CustomLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Parse custom format
        sections = content.split("---")
        
        return [
            Document(
                page_content=section.strip(),
                metadata={"source": self.file_path, "section": i}
            )
            for i, section in enumerate(sections)
            if section.strip()
        ]

# Usage
loader = CustomLoader("custom_format.txt")
docs = loader.load()
```

## Lazy Loading

Untuk file besar, gunakan lazy loading untuk hemat memori.

```python
# Regular loading - semua ke memori
docs = loader.load()

# Lazy loading - satu per satu
for doc in loader.lazy_load():
    process(doc)  # Process dan buang
```

## Metadata Enrichment

Tambahkan metadata saat loading.

```python
from langchain_community.document_loaders import TextLoader
from datetime import datetime

loader = TextLoader("document.txt")
docs = loader.load()

# Enrich metadata
for doc in docs:
    doc.metadata.update({
        "loaded_at": datetime.now().isoformat(),
        "department": "engineering",
        "doc_type": "policy"
    })
```

## Error Handling

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./documents/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    silent_errors=True  # Skip files that fail
)

docs = loader.load()
```

## Ringkasan

| Loader | Format | Install |
|--------|--------|---------|
| `TextLoader` | .txt | - |
| `PyPDFLoader` | .pdf | `pypdf` |
| `WebBaseLoader` | URLs | `beautifulsoup4` |
| `CSVLoader` | .csv | - |
| `JSONLoader` | .json | `jq` |
| `Docx2txtLoader` | .docx | `docx2txt` |
| `UnstructuredMarkdownLoader` | .md | `unstructured` |
| `YoutubeLoader` | YouTube | `youtube-transcript-api` |

**Tips:**
1. Pilih loader sesuai **format source**
2. Gunakan **lazy loading** untuk file besar
3. **Enrich metadata** untuk filtering nanti
4. Handle **errors gracefully**

---

**Selanjutnya:** [Text Splitters](/docs/rag/text-splitters) - Memecah dokumen menjadi chunks optimal.
