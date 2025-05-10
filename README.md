<h1 align="center">🧠 Knowledge Assistant</h1>
<p align="center">
A Retrieval-Augmented Q&A Chat App powered by <strong>Google Gemini</strong>, <em>designed to fetch smart answers from your documents</em>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/RAG-Powered-blueviolet?style=for-the-badge" />
</p>

---

## ✨ Features

- 🔍 **Semantic Search**: Fast, accurate retrieval using **FAISS** and **SentenceTransformers**
- 📁 **Document Upload**: Choose default content or upload your own `.txt` files
- 🤖 **Google Gemini (1.5 Flash)**: Fast, fluent LLM responses
- 🧠 **Built-in Agent Routing**: Smart query classification (`calculate`, `explain`, `define`)
- 🧮 **Calculator Agent**: Do math directly in chat!
- 💻 **Minimal UI**: Built with **Streamlit** — clean, fast, and mobile-friendly

---

## 🎬 Demo

<p align="center">
  <img src="https://github.com/yourusername/knowledge-assistant/assets/demo.gif" width="600" />
</p>

---

## 🖼️ UI Preview

<table>
<tr>
<td><img src="https://i.imgur.com/IFc1b8v.png" alt="Upload Screen" width="300"/></td>
<td><img src="https://i.imgur.com/x27NfGz.png" alt="Chat Screen" width="300"/></td>
<td><img src="https://i.imgur.com/77YqEyT.png" alt="Answer Screen" width="300"/></td>
</tr>
<tr>
<td align="center">Upload</td>
<td align="center">Ask Questions</td>
<td align="center">Get Smart Answers</td>
</tr>
</table>

---

## 🧠 How It Works

1. 📝 **Document Loading**: Custom or default `.txt` files
2. ✂️ **Chunking**: Overlapping chunks via LangChain
3. 📌 **Embedding**: Light model (`paraphrase-albert-small-v2`) via SentenceTransformers
4. 📚 **Indexing**: Chunks stored in FAISS vector DB
5. 🧠 **Query Routing**: Agent directs query to:
    - Gemini RAG
    - Calculator
6. 🧠 **LLM Response**: Gemini-1.5 Flash processes the final context

---

## 🚀 Getting Started

### 1. Clone Repo

```bash
git clone https://github.com/yourusername/knowledge-assistant.git
cd knowledge-assistant
