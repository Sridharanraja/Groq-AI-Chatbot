API='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'

import streamlit as st
import sqlite3
import json
import uuid
import os
from groq import Groq
import time
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.set_page_config(page_title="Groq AI Chatbot", page_icon="🧠")

# Database Connection
conn = sqlite3.connect("chat_database.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    chat_name TEXT,
    messages TEXT,
    model TEXT
)
""")
conn.commit()

# Initialize Groq API
client = Groq(api_key=API)

# Available models
models = {
    "Llama 3 (8B)": "llama3-8b-8192",
    "Llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "Llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "Mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "OpenAI GPT-4": "gpt-4",  # Added OpenAI
}

# Load chat history from DB
def load_chats():
    cursor.execute("SELECT * FROM chats")
    rows = cursor.fetchall()
    chats = {}
    for row in rows:
        chat_id, chat_name, messages, model = row
        chats[chat_id] = {
            "chat_name": chat_name,
            "messages": json.loads(messages),
            "model": model
        }
    return chats

# Save chat history to DB
def save_chat(chat_id, chat_name, messages, model):
    cursor.execute(
        "REPLACE INTO chats (chat_id, chat_name, messages, model) VALUES (?, ?, ?, ?)",
        (chat_id, chat_name, json.dumps(messages), model)
    )
    conn.commit()

# Delete chat from DB
def delete_chat(chat_id):
    cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    del st.session_state.chats[chat_id]
    if st.session_state.current_chat == chat_id:
        st.session_state.current_chat = list(st.session_state.chats.keys())[0] if st.session_state.chats else None

# Load all chats
chats = load_chats()

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = chats
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = None

# Function to create a new chat
def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = {"chat_name": "New Chat", "messages": [], "model": list(models.keys())[0]}
    st.session_state.current_chat = new_chat_id
    save_chat(new_chat_id, "New Chat", [], list(models.keys())[0])

# Ensure at least one chat exists
if not st.session_state.chats:
    create_new_chat()

# Sidebar for chat management
st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for chat_id, chat_data in list(st.session_state.chats.items()):
    col1, col2, col3 = st.sidebar.columns([0.7, 0.15, 0.15])

    if col1.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

    if col2.button("✎", key=f"rename_{chat_id}"):
        st.session_state.rename_mode = chat_id
        st.rerun()

    if col3.button("❌", key=f"delete_{chat_id}"):
        delete_chat(chat_id)
        st.rerun()

    if st.session_state.rename_mode == chat_id:
        new_name = st.text_input("Rename Chat:", value=chat_data["chat_name"], key=f"rename_input_{chat_id}")
        if st.button("✔️ Save", key=f"save_{chat_id}"):
            st.session_state.chats[chat_id]["chat_name"] = new_name
            save_chat(chat_id, new_name, st.session_state.chats[chat_id]["messages"], st.session_state.chats[chat_id]["model"])
            st.session_state.rename_mode = None
            st.rerun()
        if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
            st.session_state.rename_mode = None
            st.rerun()

# RAG Integration
st.title("RAG Application")

# Select folder path for document processing
folder_path = "./sheets/DATA/"

if folder_path and os.path.isdir(folder_path):
    # Get all document files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".doc", ".docx", ".odt"))]

    if files:
        # st.write(f"Found {len(files)} document(s) in the folder.")

        # Load documents
        loader = UnstructuredFileLoader(files=files)
        data = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Create a vectorstore
        vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # Initialize LLM
        llm = OpenAI(temperature=0.4, max_tokens=500)

        # Chat input
        query = st.chat_input("Say something: ")

        if query:
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})

            st.write(response["answer"])

    else:
        st.warning("No documents found in the specified folder. Please check the path and ensure it contains valid documents.")
else:
    st.warning("Please enter a valid folder path.")


