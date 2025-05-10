<h1 align="center">🧠 RAG-Powered Multi-Agent Knowledge Assistant</h1>
<p align="center">
A smart, context-aware Q&A chatbot powered by <strong>FAISS</strong>, <strong>LangChain</strong>, and <strong>Google Gemini</strong>, designed for fast, relevant, and insightful answers from custom documents.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-00897B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FAISS-00599C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" />
</p>

---

## 📌 Assignment Objective

> Build a simple “Knowledge Assistant” that:
- ✅ Retrieves relevant information using RAG
- ✅ Generates natural-language answers via an LLM
- ✅ Implements an agentic workflow that routes queries based on intent

---

## 🧠 Features

- 🔍 **Document Retrieval (RAG)**: Top-3 semantic chunks from uploaded files
- 🤖 **LLM Integration**: Uses Google Gemini 1.5 Flash API for responses
- 🧠 **Multi-Agent Routing**:
  - If query includes `calculate` → route to calculator tool
  - If query includes `define` → route to dictionary
  - Else → run RAG pipeline
- 📁 **Custom File Upload** or Use Defaults
- 🎛️ **Streamlit Web UI**: Easy interface for questions, answers, and debug logs

---

## 🧩 Architecture Overview
                ┌────────────────────────────┐
                │  Text Documents (3-5)      │
                └────────────┬───────────────┘
                             ↓
                     [Chunking & Embedding]
                             ↓
                ┌────────────────────────────┐
                │     FAISS Vector DB        │
                └────────────┬───────────────┘
                             ↓
                     [User Query Input]
                             ↓
                  [Query Router (Agentic)]
          ┌─────────────┬─────────────┬──────────────┐
          ↓             ↓             ↓              ↓
      [Calculator]  [Dictionary]   [Retriever] → [Gemini LLM]
                                                   ↓
                                           [Final Answer]

---

## 🔧 Design Choices

| Component           | Tech Used                     | Reason                                                                 |
|---------------------|-------------------------------|------------------------------------------------------------------------|
| **Vector DB**        | FAISS                         | Fast, offline, resource-light retrieval                                |
| **Embedding Model**  | `paraphrase-albert-small-v2` | Lightweight for offline use, good semantic matching                    |
| **LLM**              | Google Gemini 1.5 Flash       | Cost-effective, powerful, and multilingual                             |
| **Agent Framework**  | Manual with LangChain tools   | Simple conditional routing (calc/define) for control and flexibility   |
| **UI**               | Streamlit                     | Lightweight, interactive, no frontend required                         |

---

## 📂 Project Structure
📦 knowledge-assistant/
├── knowledge_assistant.py # Main Streamlit app
├── requirements.txt # Required packages
├── data/ # Default text documents
│ ├── file1.txt
│ ├── file2.txt
│ └── ...
├── utils/
│ └── agent_router.py # Custom query classification logic
└── README.md # This file

---

## 🛠 Setup Instructions

## 1. Clone the Repository
git clone https://github.com/yourusername/knowledge-assistant.git
cd knowledge-assistant

## 2. (Optional) Create and Activate Virtual Environment
python -m venv venv
## On Windows:
venv\Scripts\activate
## On macOS/Linux:
source venv/bin/activate

## 3. Install Dependencies
pip install -r requirements.txt

## 4. Get Your Gemini API Key
 Visit: https://aistudio.google.com/app/apikey
#Sign in and generate your API key.

## 5. Add Your Gemini API Key
 Open app.py and insert:
 import google.generativeai as genai
 genai.configure(api_key="YOUR_API_KEY_HERE")

## 6. Run the Streamlit App
streamlit run app.py

## 7. Open in Browser:
 Navigate to http://localhost:8501



