---
sidebar_position: 4
title: Manajemen API Keys
description: Cara aman mendapatkan dan mengelola API keys untuk LLM providers
---

# Manajemen API Keys & Secrets

Hampir semua LLM providers membutuhkan API key untuk autentikasi. Di bab ini, kita akan belajar cara mendapatkan dan mengelola API keys dengan aman.

:::danger Jangan Pernah Hardcode!
**Jangan pernah** menyimpan API key langsung di code. Ini adalah kesalahan keamanan yang serius dan bisa menyebabkan tagihan tak terduga jika key bocor.
:::

## Mendapatkan API Keys

### 🔵 OpenAI

1. Buka [platform.openai.com](https://platform.openai.com/)
2. Buat akun atau login
3. Klik ikon profil → **"View API Keys"**
4. Klik **"Create new secret key"**
5. Beri nama (misalnya "langchain-learning")
6. **Copy dan simpan** - key hanya ditampilkan sekali!


### 🟣 Anthropic (Claude)

1. Buka [console.anthropic.com](https://console.anthropic.com/)
2. Buat akun atau login
3. Klik **"API Keys"** di sidebar
4. Klik **"Create Key"**
5. Copy dan simpan key

### 🔴 Google (Gemini)

1. Buka [aistudio.google.com](https://aistudio.google.com/)
2. Login dengan akun Google
3. Klik **"Get API Key"**
4. Pilih project atau buat baru
5. Copy API key

### 🟠 HuggingFace

1. Buka [huggingface.co](https://huggingface.co/)
2. Buat akun atau login
3. Klik profil → **"Settings"**
4. Pilih **"Access Tokens"**
5. Klik **"New token"**
6. Pilih type "Read" atau "Write"

## Menyimpan API Keys dengan `.env`

### Setup python-dotenv

```bash
pip install python-dotenv
```

### Buat File `.env`

```bash title=".env"
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# HuggingFace
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LangSmith (opsional, untuk observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=belajar-langchain
```

### Load di Python

```python title="load_env.py"
from dotenv import load_dotenv
import os

# Load environment variables dari .env
load_dotenv()

# Sekarang bisa diakses via os.environ
openai_key = os.environ.get("OPENAI_API_KEY")
print(f"OpenAI key loaded: {openai_key[:10]}...")  # Tampilkan 10 karakter pertama saja
```

### Contoh Penggunaan dengan LangChain

```python title="example_with_env.py"
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Load .env di awal script
load_dotenv()

# LangChain akan otomatis membaca dari environment variable!
# Tidak perlu pass API key secara eksplisit

# OpenAI - membaca OPENAI_API_KEY
llm_openai = ChatOpenAI(model="gpt-4o-mini")

# Anthropic - membaca ANTHROPIC_API_KEY  
llm_claude = ChatAnthropic(model="claude-3-haiku-20240307")

# Test
response = llm_openai.invoke("Halo, apa kabar?")
print(response.content)
```

:::info Automatic Loading
LangChain akan otomatis membaca environment variables dengan nama standar:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN`
:::

## Environment Variables Naming Convention

| Provider | Environment Variable | LangChain Package |
|----------|---------------------|-------------------|
| OpenAI | `OPENAI_API_KEY` | `langchain-openai` |
| Anthropic | `ANTHROPIC_API_KEY` | `langchain-anthropic` |
| Google | `GOOGLE_API_KEY` | `langchain-google-genai` |
| HuggingFace | `HUGGINGFACEHUB_API_TOKEN` | `langchain-huggingface` |
| Cohere | `COHERE_API_KEY` | `langchain-cohere` |
| Mistral | `MISTRAL_API_KEY` | `langchain-mistralai` |

## Best Practices Keamanan

### ✅ DO (Lakukan)

```python
# 1. Gunakan environment variables
import os
api_key = os.environ.get("OPENAI_API_KEY")

# 2. Validasi key ada sebelum digunakan
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

# 3. Gunakan .env untuk development
from dotenv import load_dotenv
load_dotenv()
```

### ❌ DON'T (Jangan)

```python
# 1. JANGAN hardcode API key
api_key = "sk-1234567890abcdef"  # ❌ BERBAHAYA!

# 2. JANGAN commit .env ke git
# Pastikan .env ada di .gitignore

# 3. JANGAN print full API key
print(f"Using key: {api_key}")  # ❌ Key bisa terlihat di logs
```

### Handling Secrets di Production

```python title="config.py"
import os
from functools import lru_cache

class Settings:
    """Application settings with lazy loading."""
    
    @property
    @lru_cache()
    def openai_api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please set it in your environment or .env file."
            )
        return key
    
    @property
    @lru_cache()
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")
    
    @property
    def debug(self) -> bool:
        return os.environ.get("DEBUG", "false").lower() == "true"

settings = Settings()
```

## Multiple Environments

Untuk mengelola berbagai environment (development, staging, production):

```bash
# .env.development
OPENAI_API_KEY=sk-dev-key-xxxxx
LANGCHAIN_PROJECT=langchain-dev
DEBUG=true

# .env.production
OPENAI_API_KEY=sk-prod-key-xxxxx
LANGCHAIN_PROJECT=langchain-prod
DEBUG=false
```

Load berdasarkan environment:

```python title="load_config.py"
import os
from dotenv import load_dotenv

# Tentukan environment
env = os.environ.get("APP_ENV", "development")

# Load file .env yang sesuai
load_dotenv(f".env.{env}")

print(f"Loaded configuration for: {env}")
```

## Keamanan Tambahan

### 1. Gunakan Secret Manager di Production

Untuk production, pertimbangkan secret managers:

- **AWS Secrets Manager**
- **Google Secret Manager**
- **HashiCorp Vault**
- **Azure Key Vault**

```python
# Contoh dengan AWS Secrets Manager
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

# Usage
secrets = get_secret("langchain/api-keys")
openai_key = secrets["OPENAI_API_KEY"]
```

### 2. Rotate Keys Secara Berkala

- Set reminder untuk rotate API keys setiap 90 hari
- Jangan share keys antar project
- Revoke keys yang tidak digunakan

### 3. Set Spending Limits

Di setiap provider, set spending limit:

**OpenAI:**
1. Buka [platform.openai.com/account/limits](https://platform.openai.com/account/limits)
2. Set "Monthly budget" sesuai kebutuhan

**Anthropic:**
1. Buka Console → Settings → Billing
2. Set spending limit

## Verifikasi Setup

```python title="verify_keys.py"
#!/usr/bin/env python3
"""Verify all API keys are working."""

from dotenv import load_dotenv
import os

load_dotenv()

def check_openai():
    """Check OpenAI API key."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return "❌ Not set"
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=5)
        llm.invoke("Hi")
        return "✅ Valid"
    except Exception as e:
        return f"❌ Error: {e}"

def check_anthropic():
    """Check Anthropic API key."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "⚠️ Not set (optional)"
    
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=5)
        llm.invoke("Hi")
        return "✅ Valid"
    except Exception as e:
        return f"❌ Error: {e}"

def main():
    print("🔐 Checking API Keys...\n")
    
    print(f"OpenAI:    {check_openai()}")
    print(f"Anthropic: {check_anthropic()}")
    
    print("\n✅ Key verification complete!")

if __name__ == "__main__":
    main()
```

## Ringkasan

1. ✅ Dapatkan API keys dari provider yang diinginkan
2. ✅ Simpan di file `.env` (jangan commit ke git!)
3. ✅ Gunakan `python-dotenv` untuk load
4. ✅ Set spending limits di setiap provider
5. ✅ Verifikasi keys bekerja dengan test script

---

## 🎯 Use Case Modul 0: Hello LangChain

Setelah setup selesai, mari buat aplikasi pertama kita!

```python title="hello_langchain.py"
#!/usr/bin/env python3
"""Hello LangChain - Aplikasi pertama kita!"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Buat LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Panggil LLM pertama kali!
response = llm.invoke("Halo! Saya baru belajar LangChain. Bisakah kamu berikan motivasi singkat?")

print("🤖 Response dari LLM:")
print(response.content)
```

Jalankan:

```bash
python hello_langchain.py
```

Output:

```
🤖 Response dari LLM:
Halo! Selamat datang di dunia LangChain! 🎉

Belajar LangChain adalah langkah yang tepat untuk memahami bagaimana 
membangun aplikasi AI yang powerful. Setiap baris kode yang kamu tulis 
adalah investasi untuk masa depan karirmu.

Ingat: setiap expert pernah menjadi pemula. Teruslah eksplor, jangan 
takut error, dan nikmati prosesnya! 

Semangat! 🚀
```

**Selamat!** 🎉 Kamu sudah berhasil memanggil LLM pertamamu dengan LangChain!

---

**Selanjutnya:** [Modul 1: Fondasi LangChain](/docs/fondasi/chat-models-vs-llms) - Kita akan mempelajari building blocks dasar LangChain.
