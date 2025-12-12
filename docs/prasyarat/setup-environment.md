---
sidebar_position: 3
title: Setup Environment Python
description: Menyiapkan environment development untuk LangChain
---

# Setup Environment Python

Di bab ini, kita akan menyiapkan environment Python yang optimal untuk belajar dan mengembangkan aplikasi LangChain.

## Prasyarat

Sebelum mulai, pastikan kamu memiliki:

- **Python 3.10+** (disarankan 3.11 atau 3.12)
- **pip** (package manager Python)
- **Git** (untuk version control)
- **Text editor/IDE** (VS Code, PyCharm, atau sejenisnya)

## Instalasi Python

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# Verifikasi instalasi
python3.11 --version
```

### macOS

```bash
# Dengan Homebrew
brew install python@3.11

# Verifikasi
python3.11 --version
```

### Windows

1. Download dari [python.org](https://www.python.org/downloads/)
2. Jalankan installer
3. ✅ Centang "Add Python to PATH"
4. Pilih "Customize installation" → pastikan pip tercentang

```powershell
# Verifikasi di PowerShell
python --version
```

## Virtual Environment

:::tip Best Practice
**Selalu gunakan virtual environment!** Ini mengisolasi dependencies proyek dan mencegah konflik antar proyek.
:::

### Option 1: `venv` (Built-in Python)

```bash
# Buat project folder
mkdir belajar-langchain
cd belajar-langchain

# Buat virtual environment
python3.11 -m venv venv

# Aktivasi
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Setelah aktif, prompt akan berubah:
# (venv) $
```

### Option 2: `uv` (Recommended! Fast & Modern)

`uv` adalah package manager Python modern yang sangat cepat (ditulis dalam Rust).

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Buat project dengan uv
uv init belajar-langchain
cd belajar-langchain

# Buat virtual environment
uv venv

# Aktivasi
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install package dengan uv (jauh lebih cepat dari pip!)
uv pip install langchain langchain-openai
```

**Kenapa `uv`?**
- 10-100x lebih cepat dari pip
- Resolusi dependency yang lebih baik
- Modern tooling (mirip npm/cargo)

### Option 3: `conda` (Data Science Focus)

```bash
# Install Miniconda dari https://docs.conda.io/en/latest/miniconda.html

# Buat environment
conda create -n langchain python=3.11

# Aktivasi
conda activate langchain

# Install packages
conda install -c conda-forge langchain
# atau kombinasi conda + pip
pip install langchain-openai
```

## Struktur Project yang Disarankan

```
belajar-langchain/
├── .venv/                    # Virtual environment
├── .env                      # API keys (JANGAN commit!)
├── .gitignore               # Git ignore file
├── requirements.txt         # Dependencies
├── pyproject.toml           # Project config (modern)
├── README.md                # Project documentation
├── notebooks/               # Jupyter notebooks untuk eksperimen
│   ├── 01_basic_llm.ipynb
│   └── 02_rag_example.ipynb
├── src/                     # Source code
│   ├── __init__.py
│   ├── chains/              # Custom chains
│   ├── agents/              # Agent definitions
│   └── tools/               # Custom tools
└── tests/                   # Unit tests
    └── test_chains.py
```

## Requirements File

Buat `requirements.txt`:

```txt title="requirements.txt"
# Core LangChain
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0

# LLM Providers
langchain-openai>=0.2.0
# langchain-anthropic>=0.2.0  # Uncomment jika pakai Claude

# Vector Stores & Embeddings
chromadb>=0.4.0
# faiss-cpu>=1.7.4  # Alternative vector store

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0

# Development
jupyter>=1.0.0
ipykernel>=6.0.0
```

Install dependencies:

```bash
# Dengan pip
pip install -r requirements.txt

# Dengan uv (lebih cepat!)
uv pip install -r requirements.txt
```

## Modern: `pyproject.toml`

Untuk project yang lebih serius, gunakan `pyproject.toml`:

```toml title="pyproject.toml"
[project]
name = "belajar-langchain"
version = "0.1.0"
description = "Learning LangChain with practical examples"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "chromadb>=0.4.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "pytest>=8.0.0",
    "ruff>=0.1.0",
]

[tool.ruff]
line-length = 88
target-version = "py311"
```

Install dengan:

```bash
# Dengan pip
pip install -e ".[dev]"

# Dengan uv
uv pip install -e ".[dev]"
```

## Jupyter Notebook vs Python Script

### Jupyter Notebook

**Kelebihan:**
- Interaktif - lihat output langsung
- Bagus untuk eksperimen dan learning
- Bisa mix code, markdown, dan visualisasi

**Kekurangan:**
- Tidak cocok untuk production
- Version control kurang baik
- Debugging lebih sulit

```bash
# Install Jupyter
pip install jupyter ipykernel

# Daftarkan kernel
python -m ipykernel install --user --name=langchain

# Jalankan
jupyter notebook
```

### Python Script

**Kelebihan:**
- Production-ready
- Mudah di-test dan debug
- Version control friendly

**Kekurangan:**
- Kurang interaktif
- Perlu reload untuk setiap perubahan

**Rekomendasi:**
- 📓 **Notebook** untuk learning dan eksperimen
- 📜 **Script** untuk aplikasi production

## Konfigurasi VS Code

Jika menggunakan VS Code, install extension berikut:

```json title=".vscode/extensions.json"
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff"
  ]
}
```

Settings yang disarankan:

```json title=".vscode/settings.json"
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

## Verifikasi Instalasi

Buat file `test_setup.py`:

```python title="test_setup.py"
#!/usr/bin/env python3
"""Test script untuk verifikasi setup LangChain."""

def main():
    print("🔍 Checking Python version...")
    import sys
    print(f"   Python {sys.version}")
    
    print("\n📦 Checking LangChain installation...")
    try:
        import langchain
        print(f"   langchain: {langchain.__version__}")
    except ImportError:
        print("   ❌ langchain not installed")
        return False
    
    print("\n📦 Checking langchain-core...")
    try:
        import langchain_core
        print(f"   langchain-core: {langchain_core.__version__}")
    except ImportError:
        print("   ❌ langchain-core not installed")
        return False
    
    print("\n📦 Checking langchain-openai...")
    try:
        import langchain_openai
        print(f"   langchain-openai: ✅ installed")
    except ImportError:
        print("   ⚠️  langchain-openai not installed (optional)")
    
    print("\n📦 Checking python-dotenv...")
    try:
        import dotenv
        print(f"   python-dotenv: ✅ installed")
    except ImportError:
        print("   ❌ python-dotenv not installed")
        return False
    
    print("\n✅ Setup verified successfully!")
    print("\n🚀 Next step: Configure your API keys in .env file")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

Jalankan:

```bash
python test_setup.py
```

Output yang diharapkan:

```
🔍 Checking Python version...
   Python 3.11.x ...

📦 Checking LangChain installation...
   langchain: 0.3.x

📦 Checking langchain-core...
   langchain-core: 0.3.x

📦 Checking langchain-openai...
   langchain-openai: ✅ installed

📦 Checking python-dotenv...
   python-dotenv: ✅ installed

✅ Setup verified successfully!

🚀 Next step: Configure your API keys in .env file
```

## .gitignore

Jangan lupa buat `.gitignore`:

```gitignore title=".gitignore"
# Virtual environment
.venv/
venv/
env/

# Environment variables (PENTING!)
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Vector stores (bisa besar)
chroma_db/
*.faiss

# OS
.DS_Store
Thumbs.db
```

## Ringkasan

1. ✅ Install Python 3.10+
2. ✅ Buat virtual environment (`venv` atau `uv`)
3. ✅ Install dependencies via `requirements.txt`
4. ✅ Setup IDE (VS Code recommended)
5. ✅ Verifikasi dengan test script
6. ✅ Buat `.gitignore`

---

**Selanjutnya:** [Manajemen API Keys](/docs/prasyarat/manajemen-api-keys) - Cara aman mengelola API keys untuk berbagai LLM providers.
