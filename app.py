import os
import pandas as pd
import streamlit as st
from glob import glob

from langchain_community.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ---- CONFIG ----
DOCUMENTS_FOLDER = "chatbot_docs"
MODEL_NAME = "gpt-3.5-turbo"
MAX_ROWS_PER_FILE = 100
FAISS_INDEX_PATH = f"./faiss_index_{MODEL_NAME}"

# ---- AUTH ----
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---- INIT MODELS ----
embeddings = OpenAIEmbeddings()
chat_model = ChatOpenAI(model_name=MODEL_NAME)

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

    vectorstore = FAISS.from_documents(all_documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    st.success("Index built successfully!")
else:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# ---- BUILD QA CHAIN ----
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    memory=memory
)

# ---- CHAT UI ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something from your data...")

if user_input:
    prompt = f"""
    You are an expert advisor on traffic, air quality, and weather in Riyadh, Saudi Arabia.

    If the question is unrelated to those topics or not found in the embedded data, reply:
    "Sorry, I only answer questions related to Riyadh’s traffic, air quality, or weather."

    Do not guess or provide examples. Do not generate additional explanations.
    Answer in English only.

    Question: {user_input}
    """
    response = qa_chain.run(prompt)
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
