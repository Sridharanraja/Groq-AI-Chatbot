API='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'

import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from groq import Groq
import pandas as pd

st.set_page_config(page_title="Groq AI Chatbot", page_icon="🧠")

# Google Sheets Authentication
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("./groqbot-streamlit-1f71fc746cbd.json", scope)
client_gsheets = gspread.authorize(creds)
sheet = client_gsheets.open("ChatbotHistory").sheet1 

# Initialize Groq client
groq_client = Groq(api_key=API)

# Available Groq models
models = {
    "Llama 3 (8B)": "llama3-8b-8192",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
}

# Load previous chat history from Google Sheets
def load_chat_history():
    records = sheet.get_all_records()
    chat_sessions = {}
    for row in records:
        chat_id = row["Chat_ID"]
        if chat_id not in chat_sessions:
            chat_sessions[chat_id] = []
        chat_sessions[chat_id].append({"role": row["Role"], "content": row["Content"]})
    return chat_sessions

# Save chat messages to Google Sheets
def save_chat_to_sheets(chat_id, role, content):
    sheet.append_row([chat_id, role, content])

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = load_chat_history()
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

def create_new_chat():
    new_chat_id = f"chat_{len(st.session_state.chats) + 1}"
    st.session_state.chats[new_chat_id] = []
    st.session_state.current_chat = new_chat_id

if not st.session_state.chats:
    create_new_chat()

# Sidebar
st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for chat_id in list(st.session_state.chats.keys()):
    if st.sidebar.button(chat_id, key=chat_id):
        st.session_state.current_chat = chat_id
        st.rerun()

st.title("🧠 Groq AI Chatbot")
selected_model = st.selectbox("Choose AI Model", list(models.keys()), key="model_selection")

# Display chat history
if st.session_state.current_chat:
    chat_history = st.session_state.chats[st.session_state.current_chat]
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        chat_history.append({"role": "user", "content": user_input})
        save_chat_to_sheets(st.session_state.current_chat, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            try:
                response = groq_client.chat.completions.create(
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
        save_chat_to_sheets(st.session_state.current_chat, "assistant", bot_response)
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        chat_history.append({"role": "assistant", "content": bot_response})
        save_chat_history()  # Save after every message

        with st.chat_message("assistant"):
            st.markdown(bot_response)
