import os
import pandas as pd
import streamlit as st
from glob import glob

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ---- CONFIG ----
DOCUMENTS_FOLDER = "chatbot_docs"
MODEL_NAME = "tinyllama"
MAX_ROWS_PER_FILE = 100
FAISS_INDEX_PATH = f"./faiss_index_{MODEL_NAME}"

# ---- INIT MODELS ----
embedding_model = OllamaEmbeddings(model=MODEL_NAME)
chat_model = ChatOllama(model=MODEL_NAME)

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Ask Riyadh!", page_icon="📊")
st.title(f"📊 Ask Riyadh — powered by {MODEL_NAME}")

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

    vectorstore = FAISS.from_documents(all_documents, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    st.success("Index built successfully!")
else:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# ---- CHAT UI ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something from your data...")

if user_input:
    # Retrieve documents manually
    retrieved_docs = vectorstore.similarity_search(user_input, k=4)
    retrieved_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build prompt manually
    full_prompt = f"""
You are an expert on Riyadh's traffic, air quality, and weather.
Only answer based on the following documents. If no relevant info, say:
"Sorry, I couldn't find related info in the uploaded data."

Documents:
{retrieved_texts}

Question: {user_input}
"""

    response = chat_model.invoke(full_prompt).content
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
