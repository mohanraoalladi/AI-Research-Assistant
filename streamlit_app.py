import os
import streamlit as st
from typing import List

from utils.doc_loader import load_documents
from utils.indexer import create_vector_store
from utils.chatbot import build_chain
import settings

# ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("AI Research Assistant with Contextual Memory 📚")

# sidebar for configuration, file upload and indexing
with st.sidebar.expander("⚙️ Configuration"):
    # LLM backend selector
    options = ["llama", "openai", "gemini", "ollama"]
    default_idx = options.index(settings.LLM_TYPE) if settings.LLM_TYPE in options else 0
    llm_choice = st.selectbox(
        "Choose LLM backend", options, index=default_idx
    )
    if llm_choice == "llama":
        llama_path = st.text_input(
            "LLaMA model path (leave blank to use default if available)",
            value=settings.LLAMA_MODEL_PATH,
        )
        settings.LLM_TYPE = "llama"
        settings.LLAMA_MODEL_PATH = llama_path
    elif llm_choice == "openai":
        api_key = st.text_input(
            "OpenAI API key", type="password", value=settings.OPENAI_API_KEY
        )
        settings.LLM_TYPE = "openai"
        settings.OPENAI_API_KEY = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    elif llm_choice == "ollama":
        base_url = st.text_input(
            "Ollama server URL", value=settings.OLLAMA_BASE_URL
        )
        model_name = st.text_input(
            "Ollama model name", value=settings.OLLAMA_MODEL
        )
        settings.LLM_TYPE = "ollama"
        settings.OLLAMA_BASE_URL = base_url
        settings.OLLAMA_MODEL = model_name
    else:
        settings.LLM_TYPE = "gemini"
        gem_key = st.text_input("Google Gemini API key", type="password", value=settings.GEMINI_API_KEY)
        settings.GEMINI_API_KEY = gem_key
    # button to force rebuild when config changes
    if st.button("Apply changes / Rebuild chain"):
        st.session_state.chain = build_chain()
        st.session_state.llm_choice = llm_choice
        # track additional config for change detection
        st.session_state.llama_path = settings.LLAMA_MODEL_PATH
        st.session_state.openai_key = settings.OPENAI_API_KEY
        st.session_state.ollama_base = settings.OLLAMA_BASE_URL
        st.session_state.ollama_model = settings.OLLAMA_MODEL

with st.sidebar.expander("📁 Upload & Index Documents"):
    uploaded_files = st.file_uploader("Choose files to upload", type=["pdf", "txt", "csv"], accept_multiple_files=True)
    if st.button("Index"):
        if uploaded_files:
            paths = []
            for file in uploaded_files:
                save_path = os.path.join(settings.DATA_DIR, file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                paths.append(save_path)
            docs = load_documents(paths)
            create_vector_store(docs)
            st.success("Indexed documents successfully.")
        else:
            st.warning("No files selected.")

# main layout: two columns for chat and visualization
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat")
    # initialize chain when needed (or when LLM choice or relevant settings change)
    need_rebuild = False
    if "chain" not in st.session_state:
        need_rebuild = True
    elif st.session_state.get("llm_choice") != llm_choice:
        need_rebuild = True
    elif llm_choice == "llama" and st.session_state.get("llama_path") != settings.LLAMA_MODEL_PATH:
        need_rebuild = True
    elif llm_choice == "ollama" and (
        st.session_state.get("ollama_base") != settings.OLLAMA_BASE_URL
        or st.session_state.get("ollama_model") != settings.OLLAMA_MODEL
    ):
        need_rebuild = True

    if need_rebuild:
        st.session_state.chain = build_chain()
        st.session_state.llm_choice = llm_choice
        # remember current configuration values
        st.session_state.llama_path = settings.LLAMA_MODEL_PATH
        st.session_state.ollama_base = settings.OLLAMA_BASE_URL
        st.session_state.ollama_model = settings.OLLAMA_MODEL

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Ask a question about your documents:")
    if st.button("Send") and question:
        # get the answer from the chain
        result = st.session_state.chain({'question': question})
        answer = result.get('answer', 'No answer generated.')
        
        # clean up the answer: remove the prompt template and cruft
        # remove leading template text
        if 'Use the following pieces of context' in answer:
            answer = answer.split('Helpful Answer:')[-1].strip() if 'Helpful Answer:' in answer else answer.split('Answer:')[-1].strip()
        
        # remove repetitive phrases that degrade quality
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            # skip lines that are just repeated Q&A patterns
            if line.strip() and not line.strip().startswith('Q:') and not line.strip().startswith('Question:'):
                cleaned_lines.append(line)
        answer = '\n'.join(cleaned_lines).strip()
        
        # take only the first good paragraph or sentence block to avoid repetition
        # split on double newlines to get paragraphs
        paragraphs = answer.split('\n\n')
        if paragraphs:
            # use first 2-3 paragraphs max
            answer = '\n\n'.join(paragraphs[:3])
        
        # manually retrieve source documents using the retriever
        from utils.indexer import get_retriever
        retriever = get_retriever()
        source_docs = retriever.get_relevant_documents(question) if hasattr(retriever, 'get_relevant_documents') else []
        
        # deduplicate sources by content hash to avoid showing same content multiple times
        seen_content = set()
        unique_docs = []
        for doc in source_docs:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            content_hash = hash(content[:100])  # hash first 100 chars to detect duplicates
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        source_docs = unique_docs
        
        # display the current answer prominently
        st.markdown("---")
        st.markdown("### Answer")
        st.write(answer)
        
        # display source documents with citations
        if source_docs:
            st.markdown("### Sources")
            for i, doc in enumerate(source_docs, 1):
                # extract page number or source if available
                source_info = doc.metadata.get('source', 'Unknown source') if hasattr(doc, 'metadata') else 'Unknown'
                page_num = doc.metadata.get('page', '') if hasattr(doc, 'metadata') else ''
                if page_num:
                    source_info += f" (Page {page_num})"
                
                with st.expander(f"📄 Source {i}: {source_info}"):
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    st.text(content[:500] + "..." if len(content) > 500 else content)
        else:
            st.info("No relevant documents found.")

with col2:
    st.subheader("Visualization")
    
    # show basic statistics about the FAISS index if present
    try:
        import os
        index_path = os.path.join(settings.DATA_DIR, "index.faiss")
        
        if os.path.exists(index_path):
            # index file exists, try to load it and get stats
            from utils.indexer import get_retriever
            retriever = get_retriever()
            
            # check if it's a real retriever (has vector_store) or dummy
            if hasattr(retriever, 'vector_store'):
                store = retriever.vector_store
                if hasattr(store, 'index') and hasattr(store.index, 'ntotal'):
                    total = store.index.ntotal
                    st.metric("Vectors in Index", total)
                    st.success("✅ Index loaded successfully")
                else:
                    st.info("📊 Index created but stats unavailable")
            else:
                st.info("📊 Index file detected")
        else:
            st.warning("⚠️ No index file yet. Upload and index documents to get started.")
    except Exception as e:
        st.warning(f"⚠️ Could not load index stats: {str(e)[:50]}")
