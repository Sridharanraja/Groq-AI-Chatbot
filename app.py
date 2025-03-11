API='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'

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import sqlite3
import json
import uuid
import os
from docx import Document
from groq import Groq
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from crewai import Agent, Task
import speech_recognition as sr
# import pyttsx3
from gtts import gTTS

# Initialize Streamlit app
st.set_page_config(page_title="Groq RAG Chatbot with CrewAI", page_icon="\U0001F9E0")

# Database Connection
conn = sqlite3.connect("chat_database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT,
    chat_name TEXT,
    messages TEXT,
    model TEXT,
    agent TEXT
)
""")
conn.commit()
try:
    cursor.execute("ALTER TABLE chats ADD COLUMN agent TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass  # Column already exists

# Initialize Groq Client
client_groq = Groq(api_key=API)

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

def load_documents():
    """Load existing documents."""
    return []

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


if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None

# def retrieve_relevant_docs(query):
#     if vector_store is None:
#         return [], "No relevant documents found. FAISS vector store is not initialized."

#     try:
#         results = vector_store.similarity_search_with_score(query, k=3)
#         if results:
#             filenames = list(set([res[0].metadata.get("source", "Unknown") for res in results]))  # ✅ Remove Duplicates
#             doc_texts = "\n".join([
#                 f"**Source: {filenames[i]}**\n{results[i][0].page_content[:1000]}" for i in range(len(results))
#             ])
#             return filenames, doc_texts
#     except Exception as e:
#         st.error(f"Error retrieving documents: {e}")

#     return [], "No relevant documents found."

def text_to_speech(text):
    """Convert text to speech and play the audio in Streamlit."""
    tts = gTTS(text=text, lang="en")
    audio_file = "response.mp3"
    tts.save(audio_file)
    st.audio(audio_file, format="audio/mp3")
    os.remove(audio_file)  # Clean up after playing

def speech_to_text():
    """Capture voice input and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand. Please try again."
        except sr.RequestError:
            return "Speech recognition service is unavailable."
        except Exception as e:
            return f"Error: {str(e)}"

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

finance_agent = Agent(
    role="Finance Analyst",
    goal="Provide financial insights using relevant financial documents.",
    backstory="An AI-driven financial expert skilled in analyzing market trends and company financials."
)

marketing_agent = Agent(
    role="Marketing Expert",
    goal="Analyze market trends based on retrieved data.",
    backstory="An AI-powered market analyst specializing in consumer behavior and competitive analysis."
)

strategy_agent = Agent(
    role="Business Strategist",
    goal="Suggest strategies using available insights.",
    backstory="A strategic AI consultant that formulates innovative business strategies based on industry trends."
)

moderator_agent = Agent(
    role="Moderator",
    goal="Ensure smooth discussions and summarize key insights.",
    backstory="A neutral AI moderator ensuring productive and balanced discussions among stakeholders."
)
agents = {
    "Finance Analyst": finance_agent,
    "Marketing Expert": marketing_agent,
    "Business Strategist": strategy_agent,
    "Moderator": moderator_agent
}

# CrewAI Tasks
finance_task = Task(description="Analyze financial reports and provide insights.", agent=finance_agent,expected_output="A detailed analysis of financial trends based on retrieved documents.")
marketing_task = Task(description="Evaluate market trends and customer behaviors.", agent=marketing_agent,expected_output="An overview of current market trends with key insights.")
strategy_task = Task(description="Generate business strategies based on retrieved insights.", agent=strategy_agent,expected_output="A set of business strategies tailored to the given insights.")
moderator_task = Task(description="Summarize key discussion points and ensure smooth interaction.", agent=moderator_agent,expected_output="A concise summary of the discussion with major takeaways.")

tasks = {
    "Finance Analyst": finance_task,
    "Marketing Expert": marketing_task,
    "Business Strategist": strategy_task,
    "Moderator": moderator_task
}

# Sidebar - Agent Selection
st.sidebar.header("Select AI Agent")
# selected_agent_name = st.sidebar.radio("Choose an Agent:", list(agents.keys()))
selected_agent_name = st.sidebar.radio(
    "Choose an Agent:", list(agents.keys()), key="agent_selector"
)

st.session_state.selected_agent = selected_agent_name

# Sidebar - Chat session management for all agents
st.sidebar.title("💬 Chats")

# Ensure session state variables exist
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Stores all chat sessions
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = None
if "show_options" not in st.session_state:
    st.session_state.show_options = {}

# Ensure selected agent has a dictionary in chats
if selected_agent_name not in st.session_state.chats:
    st.session_state.chats[selected_agent_name] = {}  # Initialize empty dictionary

# If no chat exists for this agent, create a new chat session
if not st.session_state.chats[selected_agent_name]:  
    chat_id = str(uuid.uuid4())
    st.session_state.chats[selected_agent_name][chat_id] = {
        "chat_name": f"Chat with {selected_agent_name}",
        "messages": [],
        "model": "Llama 3 (8B)",
        "agent": selected_agent_name,
    }
    st.session_state.current_chat = chat_id

# Sidebar - Manage Chat Sessions for Selected Agent
st.sidebar.subheader(f"Chats for {selected_agent_name}")

# Button to create a new chat for the selected agent
if st.sidebar.button("➕ New Chat", key=f"new_chat_{selected_agent_name}"):
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[selected_agent_name][new_chat_id] = {
        "chat_name": f"Chat with {selected_agent_name} ({len(st.session_state.chats[selected_agent_name]) + 1})",
        "messages": [],
        "model": "Llama 3 (8B)",
        "agent": selected_agent_name,
    }
    st.session_state.current_chat = new_chat_id
    st.rerun()

# Ensure correct looping over chat sessions
for chat_id, chat_data in st.session_state.chats[selected_agent_name].items():
    if not isinstance(chat_data, dict):  # Fix any incorrect data types
        continue

    col1, col2 = st.sidebar.columns([0.85, 0.15])

    # Switch chat session
    if col1.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

    # More options button
    with col2:
        if st.button("⋮", key=f"options_{chat_id}", help="More options"):
            st.session_state.show_options[chat_id] = not st.session_state.show_options.get(chat_id, False)
            st.rerun()

    # Show chat options (rename, delete)
    if st.session_state.show_options.get(chat_id, False):
        with st.sidebar.expander("Options", expanded=True):
            if st.button("📝 Rename", key=f"rename_{chat_id}"):
                st.session_state.rename_mode = chat_id
                st.session_state.show_options[chat_id] = False
                st.rerun()
            if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
                del st.session_state.chats[selected_agent_name][chat_id]
                st.session_state.current_chat = None
                st.rerun()

    # Rename chat functionality
    if st.session_state.rename_mode == chat_id:
        new_name = st.text_input("Enter new name:", value=chat_data["chat_name"], key=f"input_{chat_id}")
        if st.button("✔️ Save", key=f"save_{chat_id}"):
            st.session_state.chats[selected_agent_name][chat_id]["chat_name"] = new_name
            st.session_state.rename_mode = None
            st.rerun()
        if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
            st.session_state.rename_mode = None
            st.rerun()

# Set default chat if none is selected
if st.session_state.current_chat is None and st.session_state.chats[selected_agent_name]:
    st.session_state.current_chat = list(st.session_state.chats[selected_agent_name].keys())[0]

st.sidebar.header("Upload DOCX File for RAG")
uploaded_file = st.sidebar.file_uploader("Upload a DOCX file", type=["docx"])

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_docx(file_path)
    st.sidebar.success("Vector store updated with the new DOCX file.")

# Main Chat UI
# Ensure session state stores chat history per agent
if "chats" not in st.session_state:
    st.session_state.chats = {}

# if selected_agent_name not in st.session_state.chats:
#     chat_id = str(uuid.uuid4())  # Unique chat ID per agent
#     st.session_state.chats[selected_agent_name] = {
#         "chat_id": chat_id,
#         "chat_name": f"Chat with {selected_agent_name}",
#         "messages": [],
#         "model": "Llama 3 (8B)"
#     }

# Ensure selected agent has a dictionary in chats
if selected_agent_name not in st.session_state.chats:
    st.session_state.chats[selected_agent_name] = {}  # Initialize empty dictionary

if "current_chat" not in st.session_state or st.session_state.current_chat not in st.session_state.chats[selected_agent_name]:
    chat_id = str(uuid.uuid4())  # Generate a new chat ID
    st.session_state.chats[selected_agent_name][chat_id] = {
        "chat_name": f"Chat with {selected_agent_name}",
        "messages": [],
        "model": "Llama 3 (8B)",
        "agent": selected_agent_name,
    }
    st.session_state.current_chat = chat_id

# Now safely retrieve chat data
chat_data = st.session_state.chats[selected_agent_name][st.session_state.current_chat]

# # Create a new chat session if no chat exists for this agent
# if not st.session_state.chats[selected_agent_name]:  
#     chat_id = str(uuid.uuid4())
#     st.session_state.chats[selected_agent_name][chat_id] = {
#         "chat_name": f"Chat with {selected_agent_name}",
#         "messages": [],
#         "model": "Llama 3 (8B)",
#         "agent": selected_agent_name,
#     }
#     st.session_state.current_chat = chat_id
# else:
#     # Retrieve an existing chat_id or default to the first one
#     chat_id = st.session_state.current_chat or list(st.session_state.chats[selected_agent_name].keys())[0]
#     st.session_state.current_chat = chat_id  # Ensure it's stored in session state

# Get the chat data safely
chat_data = st.session_state.chats[selected_agent_name].get(chat_id, {})

# Ensure chat_data exists before accessing messages
if "messages" not in chat_data:
    chat_data["messages"] = []

# # Display messages
# for msg in chat_data["messages"]:
#     st.chat_message(msg["role"]).markdown(msg["content"])

def get_base64_encoded_file(file_path):
    """Reads and encodes the DOCX file to Base64 for Streamlit download."""
    with open(file_path, "rb") as f:
        file_data = f.read()
    return base64.b64encode(file_data).decode()

def generate_download_links(doc_files):
    """Creates auto-download links for only DOCX files."""
    links = []
    for doc in doc_files:
        if doc.endswith(".docx"):  # Only allow DOCX files
            file_path = os.path.join(DATA_DIR, doc)
            if os.path.exists(file_path):
                encoded_file = get_base64_encoded_file(file_path)
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{encoded_file}" download="{doc}" style="text-decoration: none; color: #00A8E8; font-weight: bold;">📄 {doc}</a>'
                links.append(href)
    return "<br>".join(links)

st.title(f"\U0001F9E0 {selected_agent_name} Chatbot")

# Model selection per agent
# model_name = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(chat_data["model"]))
# chat_data["model"] = model_name

default_model = chat_data.get("model", "Llama 3 (8B)")  # Default to Llama 3 (8B) if 'model' is missing
model_name = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(default_model))
chat_data["model"] = model_name  # Ensure model selection is saved


# Display previous messages for the selected agent
for msg in chat_data["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    st.chat_message("user").markdown(user_input)
    chat_data["messages"].append({"role": "user", "content": user_input})

    # Update the database with chat history per agent
    cursor.execute("""
        INSERT INTO chats (chat_id, user_id, chat_name, messages, model, agent)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET messages = ?, agent = ? WHERE chat_id = ?
    """, (chat_id, "default_user", chat_data["chat_name"], json.dumps(chat_data["messages"]), model_name, selected_agent_name,
          json.dumps(chat_data["messages"]), selected_agent_name, chat_id))
    conn.commit()


# 🎤 Voice Input Button
if st.button("🎤 Speak"):
    user_input = speech_to_text()
    st.chat_message("user").markdown(user_input)
    with st.spinner("Thinking..."):
        relevant_docs = retrieve_relevant_docs(user_input)
        
        if relevant_docs and relevant_docs[0] and relevant_docs[1]:
            # Convert filenames into clickable download links
            source_text = "<br>".join([f'<a href="{DATA_DIR}{doc}" download style="text-decoration: none; color: #00A8E8; font-weight: bold;">{doc}</a>' for doc in relevant_docs[0]])
            data_source = f"**Data Source: Internal Data Reference Documents are** <br><br>{source_text}"
            context = relevant_docs[1]  # Get document text
        else:
            data_source = f"**Data Source: {model_name}**"
            context = "No relevant documents found. Using AI model only."
        # st.markdown(data_source, unsafe_allow_html=True)
        full_prompt = f"Agent: {selected_agent_name}\nContext:\n{context}\n\nUser Query: {user_input}"

        client, model_id = models[model_name]
        response = client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": full_prompt}], temperature=0.5, max_tokens=1500
        )
        bot_reply = response.choices[0].message.content

    chat_data["messages"].append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").markdown(bot_reply)
    # 🎤 Speak AI Response
    text_to_speech(bot_reply)

# 🔊 Listen Again Button
if st.button("🔊 Listen Again") and chat_data["messages"]:
    last_response = chat_data["messages"][-1]["content"]
    text_to_speech(last_response)


