---
sidebar_position: 1
title: Prinsip Prompt Engineering
description: Fundamental principles untuk menulis prompt yang efektif
---

# Prinsip Prompt Engineering

Prompt engineering adalah seni dan sains dalam menulis instruksi yang jelas untuk LLM. Prompt yang baik bisa meningkatkan kualitas output secara signifikan.

## Mengapa Prompt Engineering Penting?

```python
# ❌ Bad prompt
response = llm.invoke("translate")
# Output: "Translate what? Please provide text."

# ✅ Good prompt
response = llm.invoke("Translate 'Hello, how are you?' to Indonesian")
# Output: "Halo, apa kabar?"
```

Perbedaan antara prompt yang baik dan buruk bisa berarti:
- **Akurasi**: Output yang benar vs salah
- **Konsistensi**: Hasil yang predictable
- **Efisiensi**: Fewer tokens = lower cost

## 4 Prinsip Utama

### 1. 📝 Clear Instructions

Berikan instruksi yang spesifik dan tidak ambigu.

```python
# ❌ Vague
prompt = "Write about Python"

# ✅ Specific
prompt = """
Write a 200-word introduction to Python programming language.
Target audience: Complete beginners
Tone: Friendly and encouraging
Include: What Python is used for and why it's good for beginners
"""
```

**Tips:**
- Specify **what** you want
- Specify **format** (length, structure)
- Specify **tone** and **audience**

### 2. 📋 Provide Context

Berikan background information yang relevan.

```text
❌ No context:
"Fix this code"

✅ With context:
"I'm building a REST API with FastAPI.
This endpoint should return user data but throws a 500 error.
Error: KeyError when user_id doesn't exist
Please fix this code and explain the solution."
```

### 3. 🎯 Specify Output Format

Katakan dengan jelas format output yang diharapkan.

```text
❌ Unclear format:
"List programming languages"

✅ Clear format:
"List 5 programming languages with their primary use cases.
Format each as:
- **[Language]**: [Primary use case] | [Difficulty level]

Example:
- **Python**: Data Science, Web Development | Beginner-friendly"
```

### 4. 🔢 Use Examples (Few-shot)

Berikan contoh bagaimana output seharusnya terlihat.

```text
Classify the sentiment of movie reviews.

Examples:
Review: "This movie was amazing! Best film of the year!"
Sentiment: Positive

Review: "Waste of time. Terrible acting and boring plot."
Sentiment: Negative

Now classify:
Review: "Absolutely loved it! Can't wait for the sequel!"
Sentiment:
```

## Struktur Prompt yang Baik

Template prompt yang efektif:

```text
[ROLE/PERSONA]
You are an expert [role] with deep knowledge in [domain].

[CONTEXT]
[background_information]

[TASK]
[specific_task_description]

[FORMAT]
Provide your response in the following format:
[output_format]

[CONSTRAINTS]
- [constraint_1]
- [constraint_2]

[INPUT]
[user_input]
```

### Contoh Lengkap

```python
from langchain_core.prompts import ChatPromptTemplate

code_review_prompt = ChatPromptTemplate.from_template("""
You are a senior Python developer conducting a code review.

Context:
This is a pull request for a production web application.
The code should follow PEP 8 and be production-ready.

Task:
Review the following code and provide feedback.

Format your response as:
## Summary
[Brief overview]

## Issues Found
1. [Issue with severity: Critical/Warning/Info]

## Suggestions
- [Improvement suggestions]

Code to review:
{code}
""")

chain = code_review_prompt | llm | parser
result = chain.invoke(dict(code=user_code))
```

## Zero-shot vs Few-shot

### Zero-shot

Tidak ada contoh - LLM harus memahami dari instruksi saja.

```text
Extract the person's name and age from this text:
"John Smith is a 35-year-old software engineer from Seattle."

Return as JSON with keys: name, age
```

**Kapan pakai:**
- Task sederhana dan umum
- LLM sudah familiar dengan format
- Tidak perlu style khusus

### Few-shot

Memberikan contoh untuk guide output.

```text
Extract job information from descriptions.

Example 1:
Text: "Looking for a senior developer with 5 years experience. Salary: $150k"
Output: role="senior developer", experience="5 years", salary="$150k"

Example 2:
Text: "Junior designer needed, fresh graduates welcome. $50k/year"
Output: role="junior designer", experience="entry level", salary="$50k/year"

Now extract:
Text: "[input_text]"
Output:
```

**Kapan pakai:**
- Output format khusus
- Task yang complex atau niche
- Perlu konsistensi tinggi

## Teknik Spesifik Format

### Delimiter untuk Pemisahan

```text
Summarize the text delimited by triple backticks.

\`\`\`[text]\`\`\`

Summary:
```

### Numbered Steps

```text
Follow these steps to analyze the data:

Step 1: Identify the main trends
Step 2: Calculate key statistics
Step 3: Highlight anomalies
Step 4: Provide recommendations

Data: [data]
```

### Role Playing

```text
You are a patient tutor explaining concepts to a 10-year-old.
Use simple words, analogies, and avoid technical jargon.

Explain: [concept]
```

## Common Pitfalls

### ❌ Pitfall 1: Terlalu Panjang

```text
# Bad - terlalu banyak instruksi
"Please analyze this text and provide a summary. The summary should be 
comprehensive but also concise. Make sure to include all key points..."

# Good - concise dan clear
"Summarize this text for busy executives:
- 3-5 bullet points
- Key insights only
- Professional tone"
```

### ❌ Pitfall 2: Conflicting Instructions

```text
# Bad - contradictory
"Be brief but comprehensive. Keep it short but include all details."

# Good - consistent
"Provide a brief summary (max 100 words) focusing on the 3 most important points."
```

### ❌ Pitfall 3: Assuming LLM Knowledge

```text
# Bad - assumes context
"Continue from where we left off"

# Good - self-contained
"Previous summary: [previous_summary]
New information: [new_info]
Update the summary with the new information."
```

## Best Practices Checklist

- ✅ **Be specific** - Hindari ambiguitas
- ✅ **Set constraints** - Length, format, scope
- ✅ **Provide examples** - Untuk consistency
- ✅ **Define persona** - Rol model yang diinginkan
- ✅ **Structure clearly** - Use headers, bullets, numbering
- ✅ **Test iteratively** - Refine berdasarkan output
- ✅ **Document prompts** - Version control your prompts


## Ringkasan

1. **Clear instructions** - Spesifik dan tidak ambigu
2. **Provide context** - Background information relevan
3. **Specify format** - How output should look
4. **Use examples** - Few-shot untuk consistency
5. **Structure well** - Organized and readable prompts
6. **Iterate** - Test and refine

---

**Selanjutnya:** [Few-Shot Prompting](/docs/prompt-engineering/few-shot-prompting) - Teknik memberikan contoh dalam prompt.
