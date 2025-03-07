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
# st.set_page_config(page_title="Groq RAG Chatbot", page_icon="🧠")

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

# # Initialize Groq Client
# client_groq = Groq(api_key=API)

# # Available Models (Groq only)
# models = {
#     "Llama 3 (8B)": (client_groq, "llama3-8b-8192"),
#     "llama-3 (versatile)": (client_groq, "llama-3.3-70b-versatile"),
#     "llama-3 (instant)": (client_groq, "llama-3.1-8b-instant"),
#     "mixtral-8x7b-32768": (client_groq, "mixtral-8x7b-32768")
# }

# # Load and Process Documents (adapt path to your environment)
# DATA_DIR = "./sheets/DATA/"  # Update this to your folder containing docx/odt/doc files
# VECTOR_STORE_PATH = "vector_store"

# def load_documents():
#     files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf", ".docx"))]
#     documents = []
    
#     for file in files:
#         try:
#             with open(file, "r", encoding="utf-8", errors="ignore") as f:
#                 content = f.read()
#             documents.append({"source": os.path.basename(file), "content": content})
#         except Exception as e:
#             print(f"Error processing {file}: {e}")

#     return documents

# # Vectorization
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# # Load or create vector database
# if os.path.exists(VECTOR_STORE_PATH):
#     vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
# else:
#     documents = load_documents()
#     text_chunks = [(doc["source"], chunk) for doc in documents for chunk in text_splitter.split_text(doc["content"])]
#     vector_store = FAISS.from_texts(
#         [chunk[1] for chunk in text_chunks], 
#         embeddings, 
#         metadatas=[{"source": chunk[0]} for chunk in text_chunks]
#     )
#     vector_store.save_local(VECTOR_STORE_PATH)

# def retrieve_relevant_docs(query):
#     results = vector_store.similarity_search_with_score(query, k=8)  # Adjust k as needed

#     if not results:  # No relevant docs found
#         return None, None

#     doc_names = list(set(res[0].metadata["source"] for res in results if res[0] and res[0].metadata))
#     doc_texts = "\n".join([
#         f"**Source: {res[0].metadata['source']}**\n{res[0].page_content}"
#         for res in results if res[0] and res[0].metadata
#     ])

#     return doc_names, doc_texts  # Return doc names & text



# # Load existing chats
# if "chats" not in st.session_state:
#     st.session_state.chats = {}  # Initialize empty dictionary for chats

# if "current_chat" not in st.session_state or st.session_state.current_chat not in st.session_state.chats:
#     # Create a new chat if no valid current chat exists
#     chat_id = str(uuid.uuid4())
#     st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": "Llama 3 (8B)"}
#     st.session_state.current_chat = chat_id

# if "rename_mode" not in st.session_state:
#     st.session_state.rename_mode = None

# if "show_options" not in st.session_state:
#     st.session_state.show_options = {}

# # Chat Management Functions
# def create_new_chat():
#     chat_id = str(uuid.uuid4())
#     st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": "Llama 3 (8B)"}
#     cursor.execute("REPLACE INTO chats VALUES (?, ?, ?, ?, ?)", (chat_id, "user", "New Chat", json.dumps([]), "Llama 3 (8B)"))
#     conn.commit()
#     st.session_state.current_chat = chat_id

# def rename_chat(chat_id, new_name):
#     st.session_state.chats[chat_id]["chat_name"] = new_name
#     cursor.execute("UPDATE chats SET chat_name = ? WHERE chat_id = ?", (new_name, chat_id))
#     conn.commit()
#     st.session_state.rename_mode = None
#     st.rerun()

# def delete_chat(chat_id):
#     del st.session_state.chats[chat_id]
#     cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
#     conn.commit()
#     st.rerun()

# # Sidebar - Chat session management
# st.sidebar.title("💬 Chats")

# if st.sidebar.button("➕ New Chat"):
#     create_new_chat()
#     st.rerun()

# for chat_id, chat_data in list(st.session_state.chats.items()):
#     col1, col2 = st.sidebar.columns([0.85, 0.15])

#     if col1.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
#         st.session_state.current_chat = chat_id
#         st.rerun()

#     with col2:
#         if st.button("⋮", key=f"options_{chat_id}", help="More options"):
#             st.session_state.show_options[chat_id] = not st.session_state.show_options.get(chat_id, False)
#             st.rerun()

#     if st.session_state.show_options.get(chat_id, False):
#         with st.sidebar.expander("Options", expanded=True):
#             if st.button("📝 Rename", key=f"rename_{chat_id}"):
#                 st.session_state.rename_mode = chat_id
#                 st.session_state.show_options[chat_id] = False
#                 st.rerun()
#             if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
#                 delete_chat(chat_id)

#     if st.session_state.rename_mode == chat_id:
#         new_name = st.text_input("Enter new name:", value=chat_data["chat_name"], key=f"input_{chat_id}")
#         if st.button("✔️ Save", key=f"save_{chat_id}"):
#             rename_chat(chat_id, new_name)
#         if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
#             st.session_state.rename_mode = None
#             st.rerun()

# if st.session_state.current_chat is None and st.session_state.chats:
#     st.session_state.current_chat = list(st.session_state.chats.keys())[0]

# # Main Chat UI
# st.title("🧠 Groq RAG-Enhanced Chatbot")
# chat_id = st.session_state.get("current_chat", None)

# if chat_id:
#     chat_data = st.session_state.chats[chat_id]
#     model_name = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(chat_data["model"]))
#     chat_data["model"] = model_name

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

#             if relevant_docs and relevant_docs[0] and relevant_docs[1]:
#                 source_text = ", ".join(relevant_docs[0])  # Get document names
#                 data_source = f"**Data Source: Internal Data Reference Documents:** {source_text}"
#                 context = relevant_docs[1]  # Get document text
#             else:
#                 data_source = f"**Data Source: {model_name}**"
#                 context = "No relevant documents found. Using AI model only."


#             st.markdown(data_source)
#             full_prompt = f"Context:\n{context}\n\nUser Query: {user_input}"
#             client, model_id = models[model_name]
#             response = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": full_prompt}], temperature=0.5, max_tokens=256)
#             bot_reply = response.choices[0].message.content

#         chat_data["messages"].append({"role": "assistant", "content": bot_reply})
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

# Available Models
models = {
    "Llama 3 (8B)": (client_groq, "llama3-8b-8192"),
    "llama-3 (versatile)": (client_groq, "llama-3.3-70b-versatile"),
    "llama-3 (instant)": (client_groq, "llama-3.1-8b-instant"),
    "mixtral-8x7b-32768": (client_groq, "mixtral-8x7b-32768")
}

# Load and Process Documents
DATA_DIR = "./sheets/DATA/"
VECTOR_STORE_PATH = "vector_store"

def load_documents():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf", ".docx", ".doc"))]
    documents = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            documents.append({"source": os.path.basename(file), "content": content})
        except Exception as e:
            st.warning(f"Error processing {file}: {e}")
    return documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

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
    results = vector_store.similarity_search_with_score(query, k=8)
    if not results:
        return None, None
    doc_names = list(set(res[0].metadata["source"] for res in results if res[0] and res[0].metadata))
    doc_texts = "\n".join([
        f"**Source: {res[0].metadata['source']}**\n{res[0].page_content}"
        for res in results if res[0] and res[0].metadata
    ])
    return doc_names, doc_texts

# Chat Management Functions
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": "Llama 3 (8B)"}
    cursor.execute("INSERT INTO chats VALUES (?, ?, ?, ?, ?)", (chat_id, "user", "New Chat", json.dumps([]), "Llama 3 (8B)"))
    conn.commit()
    st.session_state.current_chat = chat_id

def update_chat_title(chat_id):
    first_message = st.session_state.chats[chat_id]["messages"][0]["content"] if st.session_state.chats[chat_id]["messages"] else "New Chat"
    st.session_state.chats[chat_id]["chat_name"] = first_message[:30]
    cursor.execute("UPDATE chats SET chat_name = ? WHERE chat_id = ?", (first_message[:30], chat_id))
    conn.commit()

def edit_message(chat_id, index, new_content):
    st.session_state.chats[chat_id]["messages"][index]["content"] = new_content
    cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(st.session_state.chats[chat_id]["messages"]), chat_id))
    conn.commit()
    st.rerun()

def delete_message(chat_id, index):
    del st.session_state.chats[chat_id]["messages"][index]
    cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(st.session_state.chats[chat_id]["messages"]), chat_id))
    conn.commit()
    st.rerun()

# Sidebar
st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for chat_id, chat_data in st.session_state.chats.items():
    if st.sidebar.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

st.title("🧠 Groq RAG-Enhanced Chatbot")
chat_id = st.session_state.get("current_chat")
if chat_id:
    chat_data = st.session_state.chats[chat_id]
    for i, msg in enumerate(chat_data["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "user":
                new_text = st.text_area("Edit message", msg["content"], key=f"edit_{i}")
                if st.button("Save", key=f"save_{i}"):
                    edit_message(chat_id, i, new_text)
                if st.button("Delete", key=f"delete_{i}"):
                    delete_message(chat_id, i)
    user_input = st.chat_input("Type your message...")
    if user_input:
        chat_data["messages"].append({"role": "user", "content": user_input})
        if len(chat_data["messages"]) == 1:
            update_chat_title(chat_id)
        cursor.execute("UPDATE chats SET messages = ? WHERE chat_id = ?", (json.dumps(chat_data["messages"]), chat_id))
        conn.commit()
        st.rerun()
