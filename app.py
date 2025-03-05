API='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'

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from groq import Groq
import uuid

# Firebase Setup
cred = credentials.Certificate("./FB/groq-chatbot-firebase-adminsdk-fbsvc-ab1e7d4a51.json")  # Use your actual file
firebase_admin.initialize_app(cred)
db = firestore.client()

client = Groq(api_key=API)

# Available Groq models
models = {
    "Llama 3 (8B)": "llama3-8b-8192",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
}

st.set_page_config(page_title="Groq AI Chatbot", page_icon="🧠")

# Get User ID (For Multi-User Support)
#user_id = st.experimental_user.email or str(uuid.uuid4())  # Use email if logged in, else generate random ID
#user_ref = db.collection("users").document(user_id)

if 'user_id' not in st.session_state:
    st.session_state.user_id = st.experimental_user.email if hasattr(st.experimental_user, 'email') else str(uuid.uuid4())

user_ref = db.collection("users").document(st.session_state.user_id)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "chat_names" not in st.session_state:
    st.session_state.chat_names = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = None
if "show_options" not in st.session_state:
    st.session_state.show_options = {}

# Load chats from Firestore on app start
def load_chats():
    chats = user_ref.collection("chats").stream()
    for chat in chats:
        chat_data = chat.to_dict()
        st.session_state.chats[chat.id] = chat_data.get("messages", [])
        st.session_state.chat_names[chat.id] = chat_data.get("name", "New Chat")
    if st.session_state.chats:
        st.session_state.current_chat = list(st.session_state.chats.keys())[0]

load_chats()  # Load existing chats on refresh

# Function to create a new chat
def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = []
    st.session_state.chat_names[new_chat_id] = "New Chat"
    st.session_state.current_chat = new_chat_id
    user_ref.collection("chats").document(new_chat_id).set({
        "name": "New Chat",
        "messages": []
    })

# Function to rename chat
def rename_chat(chat_id, new_name):
    if chat_id in st.session_state.chats and new_name.strip():
        st.session_state.chat_names[chat_id] = new_name
        user_ref.collection("chats").document(chat_id).update({"name": new_name})
    st.session_state.rename_mode = None
    st.rerun()

# Function to delete chat
def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        del st.session_state.chat_names[chat_id]
        user_ref.collection("chats").document(chat_id).delete()
        if st.session_state.chats:
            st.session_state.current_chat = list(st.session_state.chats.keys())[0]
        else:
            create_new_chat()
    st.rerun()

# Function to save chat to Firestore
def save_chat(chat_id):
    user_ref.collection("chats").document(chat_id).set({
        "name": st.session_state.chat_names[chat_id],
        "messages": st.session_state.chats[chat_id]
    })

# Sidebar - Chat session management
st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for chat_id, chat_name in list(st.session_state.chat_names.items()):
    col1, col2 = st.sidebar.columns([0.85, 0.15])
    if col1.button(chat_name, key=f"chat_{chat_id}"):
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
        new_name = st.text_input("Enter new name:", value=chat_name, key=f"input_{chat_id}")
        if st.button("✔️ Save", key=f"save_{chat_id}"):
            rename_chat(chat_id, new_name)
        if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
            st.session_state.rename_mode = None
            st.rerun()

st.title("🧠 Groq AI Chatbot")
if st.session_state.current_chat:
    st.subheader(f"Session: {st.session_state.chat_names[st.session_state.current_chat]}")

selected_model = st.selectbox("Choose AI Model", list(models.keys()), key="model_selection")

if st.session_state.current_chat:
    chat_id = st.session_state.current_chat
    chat_history = st.session_state.chats[chat_id]
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_input = st.chat_input("Type your message...")
    if user_input:
        chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=models[selected_model],
                    messages=chat_history,
                    temperature=1,
                    max_tokens=1024,
                    top_p=1
                )
                bot_response = response.choices[0].message.content
            except Exception as e:
                bot_response = f"⚠️ Error: {str(e)}"
        chat_history.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        save_chat(chat_id)


        chat_history.append({"role": "assistant", "content": bot_response})
        save_chat_history()  # Save after every message

        with st.chat_message("assistant"):
            st.markdown(bot_response)
