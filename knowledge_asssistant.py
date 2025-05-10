import streamlit as st
import os
import re
import tempfile
import faiss
import pickle
import google.generativeai as genai
import torch

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Set page configuration
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for a beautiful interface
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e0e7ff;
    }
    .chat-message.assistant {
        background-color: #dbeafe;
    }
    .dataset-toggle {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# 1. Load & Chunk Documents
# ========================
def load_and_chunk_documents(folder_path=None, file_paths=None):
    documents = []

    # Load from folder (default dataset)
    if folder_path and os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(folder_path, filename))
                documents.extend(loader.load())
                print(f"Loaded {filename} from default dataset")

    # Load from custom uploaded files
    if file_paths:
        for file_path in file_paths:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded custom file: {os.path.basename(file_path)}")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

# ========================
# 2. Build FAISS Index
# ========================
def build_faiss_index(chunks, model):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, texts, embeddings

# ========================
# 3. Retrieve Top Chunks
# ========================
def retrieve_top_chunks(query, model, index, texts, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# ========================
# 4. Gemini Answer (RAG)
# ========================
def rag_llm_tool(query, relevant_texts, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    context = "\n\n".join([f"Chunk {i+1}:\n{text}" for i, text in enumerate(relevant_texts)])
    prompt = f"{context}\n\nBased on the above context, answer the following question:\n{query}"

    response = model.generate_content(prompt)
    return response.text.strip()

# ========================
# 5. Calculator Tool
# ========================
def calculator_tool(expression: str):
    try:
        # Using a safer eval approach
        # Remove any non-mathematical expressions for security
        sanitized_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(sanitized_expr)
        return f"Result of calculation: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

# ========================
# 6. Agent Router
# ========================
def route_query(query, model, index, texts, api_key):
    print(f"[Agent Log] Received query: '{query}'")
    lowered_query = query.lower()

    if "calculate" in lowered_query:
        expression = re.sub(r"calculate", "", query, flags=re.IGNORECASE).strip()
        print("[Agent Log] Detected calculation. Routing to calculator tool.")
        return calculator_tool(expression)

    elif any(kw in lowered_query for kw in ["define", "what is", "explain"]):
        print("[Agent Log] Detected definition/explanation query. Routing to RAG + Gemini.")
        top_chunks = retrieve_top_chunks(query, model, index, texts)
        return rag_llm_tool(query, top_chunks, api_key)

    else:
        print("[Agent Log] General query. Routing to RAG + Gemini.")
        top_chunks = retrieve_top_chunks(query, model, index, texts)
        return rag_llm_tool(query, top_chunks, api_key)

# ========================
# Initialize Sentence Transformer Model
# ========================
@st.cache_resource
def load_sentence_transformer(model_name="paraphrase-albert-small-v2"):
    try:
        # Explicitly set device to CPU
        device = "cpu"
        print(f"Loading SentenceTransformer model '{model_name}' on {device}")
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")
        # Fallback to a simpler model if available
        try:
            print("Attempting to load fallback model 'all-MiniLM-L6-v2'")
            model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            return model
        except Exception as e2:
            st.error(f"Failed to load fallback model: {e2}")
            return None

# ========================
# Streamlit UI
# ========================
def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        # Load model with caching to avoid reloading on each rerun
        st.session_state.model = load_sentence_transformer()
        if not st.session_state.model:
            st.error("Failed to initialize the SentenceTransformer model. The app cannot function properly.")
            st.stop()
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "dataset_choice" not in st.session_state:
        st.session_state.dataset_choice = "default"
        
    # Page Header
    st.title("🧠 Knowledge Assistant")
    st.subheader("Get answers from your documents using AI")
    
    # Introduction
    with st.expander("How to use this tool", expanded=True):
        st.markdown("""
        ### Welcome to Knowledge Assistant!
        
        This tool uses artificial intelligence to answer questions based on documents:
        
        1. **Choose your dataset** - Use our default knowledge base or upload your own documents
        2. **Ask questions** in the chat box
        3. **Get intelligent answers** powered by RAG (Retrieval-Augmented Generation)
        
        Special features:
        - Ask for definitions with "what is" or "explain"
        - Do calculations with "calculate" in your query
        """)
    
    # Sidebar for document upload and dataset choice
    st.sidebar.title("Document Settings")
    
    # Dataset choice
    st.sidebar.markdown("### Choose Knowledge Source")
    dataset_choice = st.sidebar.radio(
        "Answer questions using:",
        ["Default dataset", "Custom uploaded files"],
        index=0,
        key="dataset_radio"
    )
    
    # Update session state
    st.session_state.dataset_choice = "default" if dataset_choice == "Default dataset" else "custom"
    
    # Upload custom documents
    if st.session_state.dataset_choice == "custom":
        st.sidebar.markdown("### Upload Custom Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Choose .txt files to add to knowledge base",
            accept_multiple_files=True,
            type="txt"
        )
    else:
        uploaded_files = []
    
    # Fixed API key from your original code
    api_key = "AIzaSyBJZ-jkAt1Nxzapa5Akljc_RKfuTd5qYA0"
    
    # Default dataset path - update this to a more flexible path
    default_dataset_path = os.path.join(os.getcwd(), r"knowledge_assistant_larger_dataset")
    if not os.path.exists(default_dataset_path):
        os.makedirs(default_dataset_path)
        # Create a sample file if the directory is empty
        with open(os.path.join(default_dataset_path, "sample.txt"), "w") as f:
            f.write("This is a sample document for the Knowledge Assistant.")
    
    # Process button
    if st.sidebar.button("Process Documents") or (not st.session_state.documents_processed):
        with st.spinner("Processing documents..."):
            temp_dir = None
            file_paths = []
            
            # Save uploaded files if using custom dataset
            if uploaded_files and st.session_state.dataset_choice == "custom":
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
            
            # Choose which source to use
            folder_to_use = None
            files_to_use = None
            
            if st.session_state.dataset_choice == "default":
                folder_to_use = default_dataset_path
                files_to_use = None
                st.sidebar.info("Using default dataset")
            else:  # custom
                folder_to_use = None
                files_to_use = file_paths
                if not files_to_use:
                    st.sidebar.warning("Please upload files to use custom dataset")
                    st.session_state.documents_processed = False
                    return
                st.sidebar.info("Using custom uploaded files")
            
            # Process documents
            chunks = load_and_chunk_documents(folder_path=folder_to_use, file_paths=files_to_use)
            
            if chunks:
                # Check if model is available
                if not st.session_state.model:
                    st.error("SentenceTransformer model is not available. Cannot process documents.")
                    return
                
                index, texts, embeddings = build_faiss_index(chunks, st.session_state.model)
                
                st.session_state.index = index
                st.session_state.texts = texts
                st.session_state.documents_processed = True
                
                st.success(f"✅ Successfully processed {len(chunks)} document chunks")
            else:
                st.error("No documents found to process!")
    
    # Dataset selection summary
    if st.session_state.documents_processed:
        dataset_name = "default knowledge base" if st.session_state.dataset_choice == "default" else "uploaded documents"
        st.sidebar.success(f"✅ Using {dataset_name}")
        st.sidebar.info(f"Total chunks: {len(st.session_state.texts)}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="message">
                    <b>{'You' if message['role'] == 'user' else 'Assistant'}:</b><br>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Check if documents have been processed
        if not st.session_state.documents_processed:
            response = "Please process documents first using the button in the sidebar!"
        else:
            # Generate response
            with st.spinner("Thinking..."):
                response = route_query(
                    query, 
                    st.session_state.model, 
                    st.session_state.index, 
                    st.session_state.texts, 
                    api_key
                )
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display the new messages
        st.rerun()
    
    # Display status
    if not st.session_state.documents_processed:
        st.info("👆 Process documents using the sidebar to get started")

if __name__ == "__main__":
    main()
