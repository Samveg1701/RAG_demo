import requests
import streamlit as st

# Define the URL for the backend server
BACKEND_URL = 'http://localhost:5000/ask'
print(BACKEND_URL)
st.set_page_config(page_title="LLAMA Medical RAG")

def display_messages(messages):
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(messages):
        st.write(f"{'You: ' if is_user else 'Assistant: '}{msg}")

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_text, True))

        # Send question to backend
        response = requests.post(BACKEND_URL, json={'question': user_text})
        print(response)
        if response.status_code == 200:
            agent_text = response.json()['answer']
            st.session_state["messages"].append((agent_text, False))
        else:
            st.write("Failed to get answer from server")

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""

    st.header("LLAMA Medical RAG")

    display_messages(st.session_state.get("messages", []))
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()