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


import streamlit as st
import sqlite3
import json
import uuid
import os
from docx import Document
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize Streamlit app
st.set_page_config(page_title="Groq RAG Chatbot", page_icon="🧠")

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

# Initialize Groq Client
client_groq = Groq(api_key=API)

# Available Models (Groq only)
models = {
    "Llama 3 (8B)": (client_groq, "llama3-8b-8192"),
    "llama-3 (versatile)": (client_groq, "llama-3.3-70b-versatile"),
    "llama-3 (instant)": (client_groq, "llama-3.1-8b-instant"),
    "mixtral-8x7b-32768": (client_groq, "mixtral-8x7b-32768")
}

# Load and Process Documents
DATA_DIR = "./sheets/DATA/"  # Update this to your folder containing docx/odt/doc files
VECTOR_STORE_PATH = "vector_store"

def load_documents():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf", ".docx"))]
    documents = []
    
    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            documents.append({"source": os.path.basename(file), "content": content, "path": file})
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return documents

# Vectorization
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Load or create vector database
if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    documents = load_documents()
    text_chunks = [(doc["source"], chunk) for doc in documents for chunk in text_splitter.split_text(doc["content"])]
    vector_store = FAISS.from_texts(
        [chunk[1] for chunk in text_chunks], 
        embeddings, 
        metadatas=[{"source": chunk[0]} for chunk in text_chunks]
    )
    vector_store.save_local(VECTOR_STORE_PATH)

def retrieve_relevant_docs(query):
    results = vector_store.similarity_search_with_score(query, k=3)  # Adjust k as needed

    if not results:
        return None, None, None

    doc_names = list(set(res[0].metadata["source"] for res in results if res[0] and res[0].metadata))
    doc_texts = "\n".join([
        f"**Source: {res[0].metadata['source']}**\n{res[0].page_content[:1000]}"  # Truncate text
        for res in results if res[0] and res[0].metadata
    ])
    doc_paths = [os.path.join(DATA_DIR, doc) for doc in doc_names]
    
    return doc_names, doc_texts, doc_paths

# Main Chat UI
st.title("🧠 Groq RAG-Enhanced Chatbot")
chat_id = st.session_state.get("current_chat", None)

if chat_id:
    chat_data = st.session_state.chats[chat_id]
    model_name = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(chat_data["model"]))
    chat_data["model"] = model_name

    for msg in chat_data["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        chat_data["messages"].append({"role": "user", "content": user_input})
        cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(chat_data["messages"]), chat_id))
        conn.commit()

        with st.spinner("Thinking..."):
            relevant_docs, doc_texts, doc_paths = retrieve_relevant_docs(user_input)

            if relevant_docs:
                st.markdown("**Data Source: Internal Data Reference Documents**")
                for doc_name, doc_path in zip(relevant_docs, doc_paths):
                    st.markdown(f"[🔗 {doc_name}](file://{doc_path})")
            else:
                st.markdown(f"**Data Source: {model_name}**")
                doc_texts = "No relevant documents found. Using AI model only."

            full_prompt = f"Context:\n{doc_texts}\n\nUser Query: {user_input}"
            client, model_id = models[model_name]
            response = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": full_prompt}], temperature=0.5, max_tokens=1500)
            bot_reply = response.choices[0].message.content

        chat_data["messages"].append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").markdown(bot_reply)
