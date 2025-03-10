API='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'

import streamlit as st
import sqlite3
import json
import uuid
import os
from docx import Document
from groq import Groq
# from crewai import Agent, Task
from langchain.document_loaders import Docx2txtLoader
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

# Load and Process Documents (adapt path to your environment)
DATA_DIR = "./sheets/DATA/"  # Update this to your folder containing docx/odt/doc files
VECTOR_STORE_PATH = "vector_store"

def load_documents():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf", ".docx"))]
    documents = []
    
    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            documents.append({"source": os.path.basename(file), "content": content})
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


# Vectorization
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

def load_documents():
    """Load existing documents."""
    return []

def process_docx(file_path):
    """Process DOCX file and update vector store."""
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    text_chunks = [(file_path, chunk) for chunk in text_splitter.split_text(documents[0].page_content)]
    
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts([chunk[1] for chunk in text_chunks], metadatas=[{"source": chunk[0]} for chunk in text_chunks])
    else:
        vector_store = FAISS.from_texts(
            [chunk[1] for chunk in text_chunks], 
            embeddings, 
            metadatas=[{"source": chunk[0]} for chunk in text_chunks]
        )
    
    vector_store.save_local(VECTOR_STORE_PATH)

st.sidebar.header("Upload DOCX File")
uploaded_file = st.sidebar.file_uploader("Upload a DOCX file", type=["docx"])

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    process_docx(file_path)
    st.sidebar.success("Vector store updated with the new DOCX file.")

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

    if not results:  # No relevant docs found
        return None, None

    doc_names = list(set(res[0].metadata["source"] for res in results if res[0] and res[0].metadata))
    doc_texts = "\n".join([
        f"**Source: {res[0].metadata['source']}**\n{res[0].page_content[:1000]}"  # Truncate text
        for res in results if res[0] and res[0].metadata
    ])

    return doc_names, doc_texts  # Return doc names & text

# Load existing chats
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Initialize empty dictionary for chats

if "current_chat" not in st.session_state or st.session_state.current_chat not in st.session_state.chats:
    # Create a new chat if no valid current chat exists
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": "Llama 3 (8B)"}
    st.session_state.current_chat = chat_id

if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = None

if "show_options" not in st.session_state:
    st.session_state.show_options = {}

# Chat Management Functions
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"chat_name": "New Chat", "messages": [], "model": "Llama 3 (8B)"}
    cursor.execute("REPLACE INTO chats VALUES (?, ?, ?, ?, ?)", (chat_id, "user", "New Chat", json.dumps([]), "Llama 3 (8B)"))
    conn.commit()
    st.session_state.current_chat = chat_id

def rename_chat(chat_id, new_name):
    st.session_state.chats[chat_id]["chat_name"] = new_name
    cursor.execute("UPDATE chats SET chat_name = ? WHERE chat_id = ?", (new_name, chat_id))
    conn.commit()
    st.session_state.rename_mode = None
    st.rerun()

def delete_chat(chat_id):
    del st.session_state.chats[chat_id]
    cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    st.rerun()

# Sidebar - Chat session management
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for chat_id, chat_data in list(st.session_state.chats.items()):
    col1, col2 = st.sidebar.columns([0.85, 0.15])

    if col1.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

    with col2:
        if st.button("⋮", key=f"options_{chat_id}", help="More options"):
            st.session_state.show_options[chat_id] = not st.session_state.show_options.get(chat_id, False)
            st.rerun()

    if st.session_state.show_options.get(chat_id, False):
        with st.sidebar.expander("Options", expanded=True):
            if st.button("📝 Rename", key=f"rename_{chat_id}"):
                st.session_state.rename_mode = chat_id
                st.session_state.show_options[chat_id] = False
                st.rerun()
            if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
                delete_chat(chat_id)

    if st.session_state.rename_mode == chat_id:
        new_name = st.text_input("Enter new name:", value=chat_data["chat_name"], key=f"input_{chat_id}")
        if st.button("✔️ Save", key=f"save_{chat_id}"):
            rename_chat(chat_id, new_name)
        if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
            st.session_state.rename_mode = None
            st.rerun()

# if st.session_state.current_chat is None and st.session_state.chats:
#     st.session_state.current_chat = list(st.session_state.chats.keys())[0]
# # Define AI Agents
# finance_agent = Agent(
#     role='Finance Analyst', 
#     goal='Provide financial insights and reports', 
#     backstory='Expert in financial data analysis', 
#     verbose=True
# )

# marketing_agent = Agent(
#     role='Marketing Expert', 
#     goal='Analyze market trends and provide strategies', 
#     backstory='Specialist in marketing and business growth', 
#     verbose=True
# )

# strategy_agent = Agent(
#     role='Business Strategist', 
#     goal='Develop business strategies for growth', 
#     backstory='Expert in business planning and execution', 
#     verbose=True
# )

# moderator_agent = Agent(
#     role='Moderator', 
#     goal='Ensure smooth discussions and maintain compliance', 
#     backstory='Maintains a fair and structured conversation', 
#     verbose=True
# )

# # Define Tasks
# finance_task = Task(
#     description='Analyze financial trends and summarize.', 
#     expected_output='Financial analysis summary.', 
#     agent=finance_agent
# )

# marketing_task = Task(
#     description='Provide latest marketing trends.', 
#     expected_output='Marketing insights.', 
#     agent=marketing_agent
# )

# strategy_task = Task(
#     description='Suggest business strategies.', 
#     expected_output='Strategy recommendations.', 
#     agent=strategy_agent
# )

# moderator_task = Task(
#     description='Moderate the discussion.', 
#     expected_output='Ensuring fairness.', 
#     agent=moderator_agent
# )

# def assign_task(user_input, agent_task):
#     context = retrieve_relevant_docs(user_input)
#     full_prompt = f"Context:\n{context}\n\nUser Query: {user_input}"
#     return agent_task.run()

# Main Chat UI
st.title("🧠 Groq RAG-Enhanced Chatbot")
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.header("📊 Finance Analyst")
#     finance_input = st.text_input("Ask Finance Analyst", key="finance_input")
#     if finance_input:
#         response = assign_task(finance_input, finance_task)
#         st.write(response)

# with col2:
#     st.header("📢 Marketing Expert")
#     marketing_input = st.text_input("Ask Marketing Expert", key="marketing_input")
#     if marketing_input:
#         response = assign_task(marketing_input, marketing_task)
#         st.write(response)

# with col3:
#     st.header("📈 Business Strategist")
#     strategy_input = st.text_input("Ask Business Strategist", key="strategy_input")
#     if strategy_input:
#         response = assign_task(strategy_input, strategy_task)
#         st.write(response)

# with col4:
#     st.header("🛡️ Moderator")
#     moderator_input = st.text_input("Ask Moderator", key="moderator_input")
#     if moderator_input:
#         response = assign_task(moderator_input, moderator_task)
#         st.write(response)

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
    
        # Replace with your GitHub repo details
        GITHUB_REPO_URL = DATA_DIR
       
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(user_input)
        
            if relevant_docs and relevant_docs[0] and relevant_docs[1]:
                # Convert filenames into clickable download links
                source_text = "<br>".join([f'<a href="{GITHUB_REPO_URL}{doc}" download style="text-decoration: none; color: #00A8E8; font-weight: bold;">{doc}</a>' for doc in relevant_docs[0]])
                data_source = f"**Data Source: Internal Data Reference Documents are** <br><br>{source_text}"
                context = relevant_docs[1]  # Get document text
            else:
                data_source = f"**Data Source: {model_name}**"
                context = "No relevant documents found. Using AI model only."
            
            # Display formatted markdown with clickable file links
            st.markdown(data_source, unsafe_allow_html=True)
            
            full_prompt = f"Context:\n{context}\n\nUser Query: {user_input}"
            client, model_id = models[model_name]
            response = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": full_prompt}], temperature=0.5, max_tokens=1500)
            bot_reply = response.choices[0].message.content


        chat_data["messages"].append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").markdown(bot_reply)
