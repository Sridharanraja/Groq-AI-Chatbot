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
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
}

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
if "chat_models" not in st.session_state:
    st.session_state.chat_models = {}  # Store the model used for each chat

# Function to create a new chat session
def create_new_chat():
    new_chat_id = f"chat_{len(st.session_state.chats) + 1}"
    st.session_state.chats[new_chat_id] = []
    st.session_state.chat_names[new_chat_id] = "New Chat"
    st.session_state.chat_models[new_chat_id] = list(models.keys())[0]  # Default model
    st.session_state.current_chat = new_chat_id

# Ensure at least one chat when running for the first time
if not st.session_state.chats:
    create_new_chat()

# Function to rename chat
def rename_chat(chat_id, new_name):
    if chat_id in st.session_state.chats and new_name.strip():
        st.session_state.chat_names[chat_id] = new_name
    st.session_state.rename_mode = None
    st.rerun()

# Function to delete chat
def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        del st.session_state.chat_names[chat_id]
        del st.session_state.chat_models[chat_id]

        # Update the current chat to another available chat
        if st.session_state.chats:
            st.session_state.current_chat = list(st.session_state.chats.keys())[0]
        else:
            create_new_chat()

    st.rerun()

# Sidebar - Chat session management
st.sidebar.title("💬 Chats")

# Button to start a new chat
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

# Display previous chats
for chat_id, chat_name in list(st.session_state.chat_names.items()):
    col1, col2 = st.sidebar.columns([0.85, 0.15])

    # Show chat name
    if col1.button(chat_name, key=f"chat_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

    # Show options (Three Dots ⋮)
    with col2:
        if st.button("⋮", key=f"options_{chat_id}", help="More options"):
            st.session_state.show_options[chat_id] = not st.session_state.show_options.get(chat_id, False)
            st.rerun()

    # If options are active, show Rename/Delete buttons
    if st.session_state.show_options.get(chat_id, False):
        with st.sidebar.expander("Options", expanded=True):
            if st.button("📝 Rename", key=f"rename_{chat_id}"):
                st.session_state.rename_mode = chat_id
                st.session_state.show_options[chat_id] = False
                st.rerun()
            if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
                delete_chat(chat_id)

    # Show rename input field
    if st.session_state.rename_mode == chat_id:
        new_name = st.text_input("Enter new name:", value=chat_name, key=f"input_{chat_id}")
        if st.button("✔️ Save", key=f"save_{chat_id}"):
            rename_chat(chat_id, new_name)
        if st.button("❌ Cancel", key=f"cancel_{chat_id}"):
            st.session_state.rename_mode = None
            st.rerun()

# Ensure there's an active chat
if st.session_state.current_chat is None and st.session_state.chats:
    st.session_state.current_chat = list(st.session_state.chats.keys())[0]

# Main Page UI
st.title("🧠 Groq AI Chatbot")
if st.session_state.current_chat:
    st.subheader(f"Session: {st.session_state.chat_names[st.session_state.current_chat]}")

# Ensure model selection is retained per chat
current_chat_id = st.session_state.current_chat
if current_chat_id not in st.session_state.chat_models:
    st.session_state.chat_models[current_chat_id] = list(models.keys())[0]  # Default model

# Model selection per chat
selected_model = st.selectbox(
    "Choose AI Model", list(models.keys()), 
    index=list(models.keys()).index(st.session_state.chat_models[current_chat_id]),
    key=f"model_selection_{current_chat_id}"  # Unique key per chat
)

# Store the selected model for the current chat
st.session_state.chat_models[current_chat_id] = selected_model

# Display chat history
if current_chat_id:
    chat_history = st.session_state.chats[current_chat_id]
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Assign first user message as chat name if it's a new chat
        if st.session_state.chat_names[current_chat_id] == "New Chat":
            st.session_state.chat_names[current_chat_id] = user_input[:30]  # Limit name length

        chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call Groq API using official SDK
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

        # Save and display bot response
        chat_history.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
