API='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'

import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq AI Chatbot", page_icon="🧠")


# Initialize Groq client
client = Groq(api_key=API)

# Available Groq models
models = {
    "Llama 3 (8B)": "llama3-8b-8192",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "mixtral-8x7b-32768":"mixtral-8x7b-32768",
    # "whisper-large-v3":"whisper-large-v3",
    # "whisper-large-v3-turbo":"whisper-large-v3-turbo",
    # "deepseek-r1-distill-llama-70b-specdec":"deepseek-r1-distill-llama-70b-specdec",
    # "Mistral (7B)": "mistral-7b",#not working
    # "Gemma (7B)": "gemma-7b"#not working
}

# Database Setup
def init_db():
    with sqlite3.connect("chat_history.db") as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS chats (
                            session_id TEXT,
                            chat_name TEXT,
                            chat_history TEXT
                        )''')

# Generate unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Unique ID per session

# Load user's chat history from DB
def load_chat_history():
    with sqlite3.connect("chat_history.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chat_name, chat_history FROM chats WHERE session_id=?", (st.session_state.session_id,))
        data = cursor.fetchall()
        return {row[0]: eval(row[1]) for row in data} if data else {}

# Save user's chat history to DB
def save_chat_history():
    with sqlite3.connect("chat_history.db") as conn:
        cursor = conn.cursor()
        for chat_name, history in st.session_state.chats.items():
            cursor.execute("REPLACE INTO chats (session_id, chat_name, chat_history) VALUES (?, ?, ?)", 
                           (st.session_state.session_id, chat_name, str(history)))
        conn.commit()

# Initialize DB
init_db()

# Load chat history for this session
if "chats" not in st.session_state:
    st.session_state.chats = load_chat_history()
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# Create a new chat
def create_new_chat():
    new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[new_chat_name] = []
    st.session_state.current_chat = new_chat_name
    save_chat_history()

# Ensure at least one chat
if not st.session_state.chats:
    create_new_chat()

# Sidebar - List of chats
st.sidebar.title("💬 Chats")
for chat_name in list(st.session_state.chats.keys()):
    if st.sidebar.button(chat_name, key=chat_name):
        st.session_state.current_chat = chat_name

if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

# Main Chat UI
st.title("🧠 Groq AI Chatbot")
if st.session_state.current_chat:
    st.subheader(f"Session: {st.session_state.current_chat}")

# Model Selection
selected_model = st.selectbox("Choose AI Model", list(models.keys()))

# Display chat history
if st.session_state.current_chat:
    chat_history = st.session_state.chats[st.session_state.current_chat]
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your message...")
    if user_input:
        chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
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
        save_chat_history()  # Save after every message

        with st.chat_message("assistant"):
            st.markdown(bot_response)
