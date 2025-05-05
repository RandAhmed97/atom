import os
import pandas as pd
import streamlit as st
from glob import glob

from sentence_transformers import SentenceTransformer
import faiss
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chains.llm import LLMChain

# ---- CONFIG ----
DOCUMENTS_FOLDER = "chatbot_docs"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_ROWS_PER_FILE = 100
FAISS_INDEX_PATH = f"./faiss_index_{MODEL_NAME}"

# ---- INIT EMBEDDINGS ----
embedding_model = SentenceTransformer(MODEL_NAME)

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Ask Riyadh!", page_icon="📊")
st.title("📊 Ask Riyadh — powered by MiniLM (Free)")

# ---- LOAD / CONVERT FILES ----
all_documents = []

if not os.path.exists(FAISS_INDEX_PATH):
    st.info("Building document index for the first time...")
    file_paths = glob(os.path.join(DOCUMENTS_FOLDER, "*.csv"))
    progress = st.progress(0)
    total_files = len(file_paths)

    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path).head(MAX_ROWS_PER_FILE)
            file_name = os.path.basename(file_path)
            for _, row in df.iterrows():
                row_dict = {col: str(row[col]) for col in df.columns if pd.notna(row[col])}
                row_text = f"File: {file_name}. Data: " + " | ".join([f"{k} = {v}" for k, v in row_dict.items()])
                all_documents.append(Document(page_content=row_text))
        except Exception as e:
            st.warning(f"Error loading {file_path}: {e}")
        progress.progress((i + 1) / total_files)

    texts = [doc.page_content for doc in all_documents]
    embeddings = embedding_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    vectorstore = FAISS(embedding_model, texts, index)
    vectorstore.save_local(FAISS_INDEX_PATH)
    st.success("Index built successfully!")
else:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model)

# ---- CHAT MEMORY ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- CHAT UI ----
user_input = st.chat_input("Ask something from your data...")

if user_input:
    query_embedding = embedding_model.encode([user_input])
    D, I = vectorstore.index.search(query_embedding, k=1)
    response = vectorstore.texts[I[0][0]]

    st.session_state.chat_history.append((user_input, response))

# ---- DISPLAY CHAT ----
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# ---- SIDEBAR ----
with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
