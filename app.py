API='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'

# import os
# import json
# import uuid
# import sqlite3
# import streamlit as st
# from groq import Groq
# from openai import OpenAI
# from dotenv import load_dotenv
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Load environment variables
# load_dotenv("./sheets/.env")
# GROQ_API_KEY = API
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# st.set_page_config(page_title="Groq & OpenAI RAG Chatbot", page_icon="🧠")

# # Database Connection
# conn = sqlite3.connect("chat_database.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS chats (
#     chat_id TEXT PRIMARY KEY,
#     user_id TEXT,
#     chat_name TEXT,
#     messages TEXT,
#     model TEXT
# )
# """)
# conn.commit()

# # Initialize APIs
# client_groq = Groq(api_key=GROQ_API_KEY)
# client_openai = OpenAI(api_key=OPENAI_API_KEY)

# models = {
#     "Llama 3 (8B) - Groq": (client_groq, "llama3-8b-8192"),
#     "Mixtral - Groq": (client_groq, "mixtral-8x7b-32768"),
#     "GPT-4 - OpenAI": (client_openai, "gpt-4"),
#     "GPT-3.5 - OpenAI": (client_openai, "gpt-3.5-turbo"),
# }

# # Load and Process Documents
# DATA_DIR = "./sheets/DATA/"

# def load_documents():
#     files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".docx", ".odt", ".doc"))]
#     all_text = ""
#     for file in files:
#         loader = UnstructuredFileLoader(file)
#         docs = loader.load()
#         for doc in docs:
#             all_text += doc.page_content + "\n"
#     return all_text

# # Vectorize Documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# text_chunks = text_splitter.split_text(load_documents())
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# vector_store = FAISS.from_texts(text_chunks, embeddings)

# def retrieve_relevant_docs(query):
#     return vector_store.similarity_search(query, k=3)

# # Load chats
# st.session_state.chats = {row[0]: {"chat_name": row[2], "messages": json.loads(row[3]), "model": row[4]} for row in cursor.execute("SELECT * FROM chats")}

# # Sidebar UI for Chats
# st.sidebar.title("💬 Chats")
# if st.sidebar.button("➕ New Chat"):
#     chat_id = str(uuid.uuid4())
#     st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": list(models.keys())[0]}
#     cursor.execute("REPLACE INTO chats VALUES (?, ?, ?, ?, ?)", (chat_id, "user", "New Chat", json.dumps([]), list(models.keys())[0]))
#     conn.commit()
#     st.rerun()

# for chat_id, chat_data in list(st.session_state.chats.items()):
#     if st.sidebar.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
#         st.session_state.current_chat = chat_id
#         st.rerun()

# # Main Chat UI
# st.title("🧠 RAG-Enhanced Chatbot")
# chat_id = st.session_state.get("current_chat", None)
# if chat_id:
#     chat_data = st.session_state.chats[chat_id]
#     model_name = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(chat_data["model"]))
#     chat_data["model"] = model_name

#     # Display chat history
#     for msg in chat_data["messages"]:
#         st.chat_message(msg["role"]).markdown(msg["content"])

#     user_input = st.chat_input("Type your message...")
#     if user_input:
#         st.chat_message("user").markdown(user_input)
#         chat_data["messages"].append({"role": "user", "content": user_input})
#         cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(chat_data["messages"]), chat_id))
#         conn.commit()

#         with st.spinner("Thinking..."):
#             relevant_docs = retrieve_relevant_docs(user_input)
#             context = "\n".join([doc.page_content for doc in relevant_docs])
#             full_prompt = f"Context:\n{context}\n\nUser Query: {user_input}"
#             client, model_id = models[model_name]
#             response = client.chat.completions.create(
#                 model=model_id, messages=[{"role": "user", "content": full_prompt}], temperature=0.7, max_tokens=512
#             )
#             bot_reply = response.choices[0].message.content

#         chat_data["messages"].append({"role": "assistant", "content": bot_reply})
#         cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(chat_data["messages"]), chat_id))
#         conn.commit()

#         st.chat_message("assistant").markdown(bot_reply)


import os
import json
import uuid
import sqlite3
import streamlit as st
# import pypandoc
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import os
from unstructured.partition.docx import partition_docx
from unstructured.partition.odt import partition_odt
from unstructured.partition.doc import partition_doc
from langchain_community.document_loaders import UnstructuredFileLoader

# pypandoc.download_pandoc()


# Load environment variables
load_dotenv("./sheets/.env")
# load_dotenv("D:/Everse Ai/Groq/FB/.env")
GROQ_API_KEY = API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Groq & OpenAI RAG Chatbot", page_icon="🧠")

# Database Connection
conn = sqlite3.connect("chat_database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT,
    chat_name TEXT,
    messages TEXT,
    model TEXT
)
""")
conn.commit()

# Initialize APIs
client_groq = Groq(api_key=GROQ_API_KEY)
client_openai = OpenAI(api_key=OPENAI_API_KEY)

models = {
    "Llama 3 (8B) - Groq": (client_groq, "llama3-8b-8192"),
    "Mixtral - Groq": (client_groq, "mixtral-8x7b-32768"),
    # "GPT-4 - OpenAI": (client_openai, "gpt-4"),
    # "GPT-3.5 - OpenAI": (client_openai, "gpt-3.5-turbo"),
}

# Load and Process Documents
DATA_DIR = "./sheets/DATA/"

# def load_documents():
#     files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".docx", ".odt", ".doc"))]
#     all_text = ""
#     for file in files:
#         loader = UnstructuredFileLoader(file)
#         docs = loader.load()
#         for doc in docs:
#             all_text += doc.page_content + "\n"
#     return all_text

def load_documents():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".docx", ".odt", ".doc"))]
    all_text = ""

    for file in files:
        try:
            if file.endswith(".docx"):
                elements = partition_docx(filename=file)
            elif file.endswith(".odt"):
                elements = partition_odt(filename=file)
            elif file.endswith(".doc"):
                elements = partition_doc(filename=file)
            
            all_text += "\n".join([elem.text for elem in elements if elem.text]) + "\n"
        
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return all_text

# # Vectorize Documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# text_chunks = text_splitter.split_text(load_documents())
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# vector_store = FAISS.from_texts(text_chunks, embeddings)



# Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(load_documents())

# Use Hugging Face Embeddings (bge-m3 model)
embedding_model = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Store embeddings in FAISS
vector_store = FAISS.from_texts(text_chunks, embeddings)



def retrieve_relevant_docs(query):
    return vector_store.similarity_search(query, k=3)

# Load chats
st.session_state.chats = {row[0]: {"chat_name": row[2], "messages": json.loads(row[3]), "model": row[4]} for row in cursor.execute("SELECT * FROM chats")}

# Sidebar UI for Chats
st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": list(models.keys())[0]}
    cursor.execute("REPLACE INTO chats VALUES (?, ?, ?, ?, ?)", (chat_id, "user", "New Chat", json.dumps([]), list(models.keys())[0]))
    conn.commit()
    st.rerun()

for chat_id, chat_data in list(st.session_state.chats.items()):
    if st.sidebar.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

# Main Chat UI
st.title("🧠 RAG-Enhanced Chatbot")
chat_id = st.session_state.get("current_chat", None)
if chat_id:
    chat_data = st.session_state.chats[chat_id]
    model_name = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(chat_data["model"]))
    chat_data["model"] = model_name

    # Display chat history
    for msg in chat_data["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        chat_data["messages"].append({"role": "user", "content": user_input})
        cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(chat_data["messages"]), chat_id))
        conn.commit()

        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(user_input)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            full_prompt = f"Context:\n{context}\n\nUser Query: {user_input}"
            client, model_id = models[model_name]
            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": full_prompt}], temperature=0.7, max_tokens=512
            )
            bot_reply = response.choices[0].message.content

        chat_data["messages"].append({"role": "assistant", "content": bot_reply})
        cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(chat_data["messages"]), chat_id))
        conn.commit()

        st.chat_message("assistant").markdown(bot_reply)
