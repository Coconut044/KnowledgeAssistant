# 🧠 Knowledge Assistant

<div align="center">
  
  ![GitHub stars](https://img.shields.io/github/stars/yourusername/knowledge-assistant?style=social)
  ![GitHub forks](https://img.shields.io/github/forks/yourusername/knowledge-assistant?style=social)
  ![GitHub watchers](https://img.shields.io/github/watchers/yourusername/knowledge-assistant?style=social)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  <img src="https://raw.githubusercontent.com/yourusername/knowledge-assistant/main/docs/banner.png" alt="Knowledge Assistant Banner" width="600px"/>

  <h3>An AI-powered tool for getting intelligent answers from your documents</h3>

  <p>
    <b>Knowledge Assistant</b> uses state-of-the-art AI to help you extract answers and insights from your document collection. Powered by Retrieval-Augmented Generation (RAG), it provides precise and contextually relevant answers to your questions.
  </p>

</div>

---

## ✨ Features

- **🔍 Intelligent Question Answering** - Ask questions in natural language and get relevant answers
- **📚 Multiple Dataset Support** - Use the built-in knowledge base or upload your own documents
- **🔄 RAG (Retrieval-Augmented Generation)** - Combines document retrieval with AI-generated responses
- **💡 Special Commands**:
  - Get definitions with "what is" or "explain"
  - Perform calculations with "calculate"
- **🎨 Beautiful UI** - Clean, intuitive interface for smooth interaction
- **🔧 Flexible Configuration** - Use default settings or customize to your needs

## 🖼️ Screenshots

<div align="center">
  <img src="https://raw.githubusercontent.com/yourusername/knowledge-assistant/main/docs/screenshot1.png" alt="Knowledge Assistant Interface" width="80%"/>
  <p><i>Main interface of Knowledge Assistant with document selection and chat</i></p>
  
  <img src="https://raw.githubusercontent.com/yourusername/knowledge-assistant/main/docs/screenshot2.png" alt="Knowledge Assistant Example Usage" width="80%"/>
  <p><i>Example of asking questions and getting answers from documents</i></p>
</div>

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/knowledge-assistant.git
   cd knowledge-assistant
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key** (optional - the app includes a demo key)
   ```bash
   # Create a .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

### Running the Application

```bash
streamlit run knowledge_assistant.py
```

The app will open in your default web browser at `http://localhost:8501`.

## 💡 How to Use

1. **Choose your knowledge source**:
   - **Default dataset**: Use the built-in knowledge base
   - **Custom uploaded files**: Upload your own text documents
   - **Custom path**: Specify a folder path containing your documents

2. **Process documents**:
   - Click the "Process Documents" button in the sidebar

3. **Ask questions**:
   - Type your question in the chat input
   - Use special prefixes for specific operations:
     - "What is..." or "Explain..." for definitions
     - "Calculate..." for calculations

4. **View responses**:
   - Answers appear in the chat interface
   - The system highlights which document chunks were used to generate the response

## 🧰 Technical Details

### Architecture

Knowledge Assistant uses a RAG (Retrieval-Augmented Generation) architecture with these components:

1. **Document Processing Pipeline**:
   - Text extraction from documents
   - Chunking with `CharacterTextSplitter`
   - Semantic embedding with `SentenceTransformer`
   - Vector storage in FAISS index

2. **Retrieval System**:
   - Query embedding and semantic search
   - Top-k relevant document chunk selection

3. **Answer Generation**:
   - Context-enriched prompt construction
   - Google Gemini 1.5 Flash for answer generation

4. **Tool Router**:
   - Query classification for specialized handling
   - Custom tools for calculations and definitions

### Technologies Used

- **Streamlit**: Frontend framework
- **LangChain**: Document processing
- **SentenceTransformer**: Text embeddings
- **FAISS**: Vector similarity search
- **Google Generative AI (Gemini)**: Large Language Model

## 📊 Performance

Knowledge Assistant is optimized for:

- **Speed**: Fast document processing and query response
- **Accuracy**: High-quality, contextually relevant answers
- **Resource Efficiency**: Works well on CPU-only environments

## 🛠️ Customization

### Changing the Embedding Model

The app uses `paraphrase-albert-small-v2` by default, but you can switch to other models:

```python
# In the load_sentence_transformer function
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
```

### Adjusting Chunk Size

For different document types, you may want to adjust the chunk size:

```python
# In the load_and_chunk_documents function
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Larger chunks
```

### Using a Different LLM

You can replace Gemini with another model:

```python
# In the rag_llm_tool function
# Example for using a different model or API
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 💖 Acknowledgments

- [Streamlit](https://streamlit.io/) for the wonderful UI framework
- [LangChain](https://langchain.readthedocs.io/) for document processing tools
- [SentenceTransformer](https://www.sbert.net/) for text embeddings
- [Google Generative AI](https://ai.google.dev/) for the Gemini model

---

<div align="center">
  <p>Made with ❤️ by Your Name</p>
  <p>
    <a href="https://twitter.com/yourusername">Twitter</a> •
    <a href="https://linkedin.com/in/yourusername">LinkedIn</a> •
    <a href="https://yourusername.github.io">Website</a>
  </p>
</div>
