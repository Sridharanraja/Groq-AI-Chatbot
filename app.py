API='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'

import streamlit as st
import sqlite3
import json
import uuid
from groq import Groq

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

# Load all chats
chats = load_chats()

# # Initialize session state
# if "chats" not in st.session_state:
#     st.session_state.chats = chats
# if "current_chat" not in st.session_state:
#     st.session_state.current_chat = None
# if "rename_mode" not in st.session_state:
#     st.session_state.rename_mode = None

# Initialize session state variables
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


# Function to create a new chat
# def create_new_chat():
#     new_chat_id = str(uuid.uuid4())
#     new_chat_name = "New Chat"
#     st.session_state.chats[new_chat_id] = {"chat_name": new_chat_name, "messages": [], "model": list(models.keys())[0]}
#     st.session_state.current_chat = new_chat_id
    def create_new_chat():
    new_chat_id = f"chat_{len(st.session_state.chats) + 1}"
    st.session_state.chats[new_chat_id] = []
    st.session_state.chat_names[new_chat_id] = "New Chat"  # ✅ Set default name
    st.session_state.current_chat = new_chat_id
    # save_chat(new_chat_id, new_chat_name, [], list(models.keys())[0])

# Ensure at least one chat exists
if not st.session_state.chats:
    create_new_chat()

# Sidebar for chat management
st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for chat_id, chat_data in st.session_state.chats.items():
    col1, col2 = st.sidebar.columns([0.85, 0.15])

    if col1.button(chat_data["chat_name"], key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

    if col2.button("⋮", key=f"options_{chat_id}"):
        st.session_state.rename_mode = chat_id
        st.rerun()

    if st.session_state.rename_mode == chat_id:
        new_name = st.text_input("Rename Chat:", value=chat_data["chat_name"], key=f"rename_{chat_id}")
        if st.button("✔️ Save", key=f"save_{chat_id}"):
            st.session_state.chats[chat_id]["chat_name"] = new_name
            save_chat(chat_id, new_name, st.session_state.chats[chat_id]["messages"], st.session_state.chats[chat_id]["model"])
            st.session_state.rename_mode = None
            st.rerun()
        if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
            st.session_state.rename_mode = None
            st.rerun()

if st.session_state.current_chat is None:
    st.session_state.current_chat = list(st.session_state.chats.keys())[0]

# Main Chat UI
st.title("🧠 Groq AI Chatbot")
chat_id = st.session_state.current_chat
chat_data = st.session_state.chats[chat_id]

st.subheader(f"Session: {chat_data['chat_name']}")

# Model selection (Saved per chat)
selected_model = st.selectbox("Choose AI Model", list(models.keys()), index=list(models.keys()).index(chat_data["model"]), key=f"model_{chat_id}")
st.session_state.chats[chat_id]["model"] = selected_model
save_chat(chat_id, chat_data["chat_name"], chat_data["messages"], selected_model)

# Display chat history
for message in chat_data["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
# user_input = st.chat_input("Type your message...")
# if user_input:
#     chat_data["messages"].append({"role": "user", "content": user_input})
#     save_chat(chat_id, chat_data["chat_name"], chat_data["messages"], selected_model)

#     with st.chat_message("user"):
#         st.markdown(user_input)
# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # # If it's a new chat and still named "New Chat", update it based on the first message
    # if st.session_state.chat_names[st.session_state.current_chat] == "New Chat":
    #     # Extract a meaningful title from the first message (taking first 3 words)
    #     extracted_title = " ".join(user_input.split()[:3]).capitalize()
    #     st.session_state.chat_names[st.session_state.current_chat] = extracted_title
    if (
        st.session_state.current_chat in st.session_state.chat_names
        and st.session_state.chat_names[st.session_state.current_chat] == "New Chat"
    ):
        extracted_title = " ".join(user_input.split()[:3]).capitalize()  # Take first 3 words
        st.session_state.chat_names[st.session_state.current_chat] = extracted_title

    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model=models[selected_model],
                messages=chat_data["messages"],
                temperature=1,
                max_tokens=1024,
                top_p=1
            )
            bot_response = response.choices[0].message.content
        except Exception as e:
            bot_response = f"⚠️ Error: {str(e)}"

    chat_data["messages"].append({"role": "assistant", "content": bot_response})
    save_chat(chat_id, chat_data["chat_name"], chat_data["messages"], selected_model)

    with st.chat_message("assistant"):
        st.markdown(bot_response)
