---
sidebar_position: 5
title: Menggabungkan Komponen
description: Preview bagaimana komponen LangChain bekerja bersama dengan LCEL
---

# Menggabungkan Komponen

Di bab ini, kita akan melihat bagaimana semua komponen yang sudah dipelajari bekerja bersama dalam sebuah "chain" menggunakan LCEL (LangChain Expression Language).

## Konsep Chain

Chain adalah sekuens komponen yang dijalankan berurutan:

```
Input → Prompt Template → LLM → Output Parser → Output
```

### Chain Sederhana

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Komponen individual
prompt = ChatPromptTemplate.from_template("Jelaskan {topik} dalam 1 paragraf")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Gabungkan dengan pipe operator
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"topik": "quantum computing"})
print(result)
```

## Pipe Operator `|`

Di LangChain, `|` adalah operator untuk menghubungkan komponen:

```python
# Ini...
chain = prompt | llm | parser

# Sama dengan ini (secara konseptual)
chain = sequence(prompt, llm, parser)

# Yang terjadi saat invoke:
# 1. prompt.invoke({"topik": "..."}) → messages
# 2. llm.invoke(messages) → AIMessage
# 3. parser.invoke(AIMessage) → string
```

## Contoh Chains

### 1. Translation Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah penerjemah profesional."),
        ("human", "Terjemahkan ke {bahasa}: {teks}")
    ])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

result = translate_chain.invoke({
    "bahasa": "Bahasa Jepang",
    "teks": "Selamat pagi, apa kabar?"
})
print(result)
# おはようございます、お元気ですか？
```

### 2. Summarization Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

summarize_chain = (
    ChatPromptTemplate.from_template("""
    Ringkas artikel berikut dalam {jumlah_kalimat} kalimat:
    
    {artikel}
    
    Ringkasan:
    """)
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

article = """
Python adalah bahasa pemrograman yang dikembangkan oleh Guido van Rossum
dan pertama kali dirilis pada tahun 1991. Python dirancang dengan filosofi
menekankan keterbacaan kode, membuatnya mudah dipelajari bagi pemula.
Python mendukung berbagai paradigma pemrograman termasuk prosedural,
berorientasi objek, dan fungsional. Python banyak digunakan dalam berbagai
bidang seperti web development, data science, machine learning, dan automation.
"""

result = summarize_chain.invoke({
    "artikel": article,
    "jumlah_kalimat": 2
})
print(result)
```

### 3. Structured Extraction Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class MovieInfo(BaseModel):
    title: str = Field(description="Judul film")
    year: int = Field(description="Tahun rilis")
    genre: list[str] = Field(description="Genre film")
    rating: float = Field(description="Rating 1-10")

parser = PydanticOutputParser(pydantic_object=MovieInfo)

extract_chain = (
    ChatPromptTemplate.from_template("""
    Extract informasi film dari review berikut:
    
    {format_instructions}
    
    Review: {review}
    """).partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | parser
)

result = extract_chain.invoke({
    "review": """
    Inception (2010) adalah masterpiece dari Christopher Nolan. 
    Film sci-fi thriller ini mendapat rating 8.8 di IMDB.
    Ceritanya tentang pencurian dalam mimpi sangat orisinal.
    """
})

print(f"Title: {result.title}")
print(f"Year: {result.year}")
print(f"Genre: {result.genre}")
print(f"Rating: {result.rating}")
```

## RunnablePassthrough

Untuk meneruskan input tanpa modifikasi.

```python
from langchain_core.runnables import RunnablePassthrough

# Passthrough meneruskan input ke step berikutnya
chain = (
    {"question": RunnablePassthrough()}  # Terima string, wrap jadi dict
    | ChatPromptTemplate.from_template("Answer: {question}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Bisa langsung pass string
result = chain.invoke("What is 2+2?")
```

## RunnableLambda

Menjalankan custom Python function dalam chain.

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

def add_emoji(text: str) -> str:
    return f"🎉 {text} 🎉"

chain = (
    ChatPromptTemplate.from_template("Say hello to {name}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
    | RunnableLambda(uppercase)
    | RunnableLambda(add_emoji)
)

result = chain.invoke({"name": "World"})
print(result)
# 🎉 HELLO, WORLD! 🎉
```

## RunnableParallel

Menjalankan beberapa chains secara paralel.

```python
from langchain_core.runnables import RunnableParallel

# Define individual chains
translate_to_english = (
    ChatPromptTemplate.from_template("Translate to English: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

translate_to_french = (
    ChatPromptTemplate.from_template("Translate to French: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Run in parallel
parallel_chain = RunnableParallel(
    english=translate_to_english,
    french=translate_to_french
)

result = parallel_chain.invoke({"text": "Selamat pagi"})
print(result)
# {'english': 'Good morning', 'french': 'Bonjour'}
```

## Chain Methods

Setiap chain memiliki methods standar:

```python
# Invoke - single call
result = chain.invoke({"input": "test"})

# Stream - streaming output
for chunk in chain.stream({"input": "test"}):
    print(chunk, end="")

# Batch - multiple inputs
results = chain.batch([
    {"input": "test1"},
    {"input": "test2"}
])

# Async variants
result = await chain.ainvoke({"input": "test"})
async for chunk in chain.astream({"input": "test"}):
    print(chunk)
```

## Debugging Chains

### Verbose Logging

```python
from langchain.globals import set_verbose, set_debug

# Enable verbose logging
set_verbose(True)

# Or full debug (very detailed)
set_debug(True)

# Now run chain - will see all steps
result = chain.invoke({"topik": "AI"})
```

### Inspect Chain Structure

```python
# Lihat input schema
print(chain.input_schema.schema())

# Lihat output schema
print(chain.output_schema.schema())

# Visualize chain (di Jupyter)
chain.get_graph().print_ascii()
```

## Use Case: Translator Bot

Mari gabungkan semua konsep dalam satu aplikasi lengkap:

```python
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Output schema
class Translation(BaseModel):
    original: str = Field(description="Teks asli")
    detected_language: str = Field(description="Bahasa terdeteksi")
    english: str = Field(description="Terjemahan ke Bahasa Inggris")
    confidence: float = Field(description="Confidence score 0-1")

parser = PydanticOutputParser(pydantic_object=Translation)

# Main translation chain
translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """
        Kamu adalah penerjemah profesional multilingual.
        Analisis bahasa sumber dan terjemahkan ke Bahasa Inggris.
        """),
        ("human", """
        Terjemahkan teks berikut:
        
        {format_instructions}
        
        Teks: {text}
        """)
    ]).partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | parser
)

# Alternative languages chain (parallel)
alt_languages_chain = RunnableParallel(
    japanese=ChatPromptTemplate.from_template("Translate to Japanese: {text}") 
             | ChatOpenAI(model="gpt-4o-mini") 
             | StrOutputParser(),
    korean=ChatPromptTemplate.from_template("Translate to Korean: {text}") 
           | ChatOpenAI(model="gpt-4o-mini") 
           | StrOutputParser(),
)

# Full pipeline
def translate_with_alternatives(text: str) -> dict:
    # Main translation
    main = translate_chain.invoke({"text": text})
    
    # Get alternatives based on English translation
    alternatives = alt_languages_chain.invoke({"text": main.english})
    
    return {
        "original": text,
        "detected_language": main.detected_language,
        "english": main.english,
        "confidence": main.confidence,
        "alternatives": alternatives
    }

# Test
result = translate_with_alternatives("Selamat pagi, apa kabar?")

print(f"Original: {result['original']}")
print(f"Detected: {result['detected_language']}")
print(f"English: {result['english']}")
print(f"Japanese: {result['alternatives']['japanese']}")
print(f"Korean: {result['alternatives']['korean']}")
print(f"Confidence: {result['confidence']:.0%}")
```

Output:

```
Original: Selamat pagi, apa kabar?
Detected: Indonesian
English: Good morning, how are you?
Japanese: おはようございます、お元気ですか？
Korean: 안녕하세요, 잘 지내세요?
Confidence: 95%
```

## Preview: LCEL di Modul Berikutnya

Di Modul 2, kita akan deep dive ke LCEL dengan:

- RunnableBranch untuk conditional logic
- Error handling dengan fallbacks
- Streaming dengan event handlers
- Advanced composition patterns

## Ringkasan

1. **Chain** = sekuens komponen yang dijalankan berurutan
2. **Pipe operator `|`** menghubungkan komponen
3. **RunnablePassthrough** - meneruskan input
4. **RunnableLambda** - custom Python functions
5. **RunnableParallel** - jalankan chains secara paralel
6. Gunakan **debug mode** untuk troubleshooting

---

## 🎯 Use Case Modul 1: Complete Translator Bot

```python
#!/usr/bin/env python3
"""
Translator Bot - Use Case Modul 1
Menerjemahkan teks ke berbagai bahasa dengan format terstruktur.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class TranslationResult(BaseModel):
    """Hasil terjemahan terstruktur."""
    source_language: str = Field(description="Bahasa asal terdeteksi")
    source_text: str = Field(description="Teks asli")
    target_language: str = Field(description="Bahasa target")
    translated_text: str = Field(description="Hasil terjemahan")
    notes: str = Field(description="Catatan tentang terjemahan (idiom, konteks, dll)")

def create_translator(target_language: str = "English"):
    """Create a translator chain for specific target language."""
    parser = PydanticOutputParser(pydantic_object=TranslationResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        Kamu adalah penerjemah profesional ke {target_language}.
        
        Tugas:
        1. Deteksi bahasa sumber
        2. Terjemahkan ke {target_language}
        3. Berikan catatan jika ada idiom atau konteks khusus
        
        {{format_instructions}}
        """.format(format_instructions=parser.get_format_instructions())),
        ("human", "{text}")
    ])
    
    return (
        prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | parser
    )

def main():
    print("🌍 Translator Bot - Modul 1 Use Case\n")
    
    # Create translators
    to_english = create_translator("English")
    to_japanese = create_translator("Japanese")
    to_korean = create_translator("Korean")
    
    # Test texts
    texts = [
        "Tak kenal maka tak sayang",
        "Sedikit demi sedikit, lama-lama menjadi bukit",
        "Selamat pagi! Semoga harimu menyenangkan."
    ]
    
    for text in texts:
        print(f"📝 Original: {text}")
        print("-" * 50)
        
        # Translate to English
        result = to_english.invoke({"text": text})
        print(f"🇬🇧 English: {result.translated_text}")
        if result.notes:
            print(f"   📌 Note: {result.notes}")
        
        # Translate to Japanese
        result = to_japanese.invoke({"text": text})
        print(f"🇯🇵 Japanese: {result.translated_text}")
        
        # Translate to Korean
        result = to_korean.invoke({"text": text})
        print(f"🇰🇷 Korean: {result.translated_text}")
        
        print("\n")

if __name__ == "__main__":
    main()
```

**Selamat!** 🎉 Kamu sudah menyelesaikan Modul 1: Fondasi LangChain!

---

**Selanjutnya:** [Modul 2: LCEL](/docs/lcel/filosofi-lcel) - Deep dive ke LangChain Expression Language.
