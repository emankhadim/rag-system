import streamlit as st
import requests
from typing import Optional, Dict, List
import os

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat", layout="centered")

st.markdown("""
<style>
    .main { max-width: 800px; }
    .user-message {
        background-color: #007AFF;
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #E9ECEF;
        color: #000;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 20%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTextInput > div > div > input { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)


def check_health() -> Optional[Dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def query_system(question: str, top_k: int = 4) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": question, "top_k": top_k, "return_sources": True},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def ingest_documents(documents: List[Dict]) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"documents": documents},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.title("RAG Chat")
    
    health = check_health()
    if not health:
        st.error("API not running")
        st.stop()
    
    tab1, tab2 = st.tabs(["Chat", "Upload"])
    
    with tab2:
        st.subheader("Add Documents")
        
        doc_id = st.text_input("ID", placeholder="doc_001")
        doc_title = st.text_input("Title", placeholder="My Document")
        doc_text = st.text_area("Content", placeholder="Paste text here", height=200)
        
        if st.button("Upload", type="primary"):
            if doc_id and doc_title and doc_text:
                with st.spinner("Uploading..."):
                    result = ingest_documents([{
                        "id": doc_id,
                        "title": doc_title,
                        "text": doc_text,
                        "metadata": {}
                    }])
                    
                    if result and "error" not in result:
                        st.success("Uploaded")
                    else:
                        st.error(f"Failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Fill all fields")
    
    with tab1:
        health = check_health()
        if not health.get("vector_db_loaded"):
            st.warning("No documents loaded")
        else:
            with st.expander("Info", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", "Ready")
                    st.metric("Documents", health.get("num_documents", 0))
                    st.caption(f"{health.get('num_vectors', 0)} chunks")
                with col2:
                    st.text("Model")
                    st.caption(health.get("llm_model_name", ""))
            
            st.divider()
            
            for msg in st.session_state.messages:
                st.markdown(f'<div class="user-message">{msg["question"]}</div>', unsafe_allow_html=True)
                
                answer = msg.get("answer", "")
                if answer:
                    st.markdown(f'<div class="bot-message">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="bot-message">No answer</div>', unsafe_allow_html=True)
                
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"{len(sources)} sources", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"**{i}. {src.get('title')}** ({src.get('score', 0):.3f})")
                            st.text(src.get('text', '')[:150] + "...")
                
                st.write("")
            
            st.divider()
            
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    user_input = st.text_input(
                        "Message",
                        placeholder="Ask a question",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    send_button = st.button("Send", type="primary", use_container_width=True)
            
            if send_button and user_input:
                with st.spinner("..."):
                    result = query_system(user_input)
                    
                    if result and "error" not in result:
                        st.session_state.messages.append({
                            "question": user_input,
                            "answer": result.get("answer", ""),
                            "sources": result.get("sources", [])
                        })
                    else:
                        st.session_state.messages.append({
                            "question": user_input,
                            "answer": f"Error: {result.get('error', 'Unknown')}",
                            "sources": []
                        })
                
                st.rerun()
            
            if not st.session_state.messages:
                st.caption("Examples:")
                cols = st.columns(3)
                examples = ["What is AI?", "AI in healthcare?", "AI challenges?"]
                
                for col, example in zip(cols, examples):
                    with col:
                        if st.button(example, use_container_width=True):
                            result = query_system(example)
                            if result and "error" not in result:
                                st.session_state.messages.append({
                                    "question": example,
                                    "answer": result.get("answer", ""),
                                    "sources": result.get("sources", [])
                                })
                            st.rerun()


if __name__ == "__main__":
    main()