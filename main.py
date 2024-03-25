import streamlit as st
import google.generativeai as genai
import os
import dotenv
import chat
dotenv.load_dotenv()

st.title("RAG with GEMINI")

uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    with open(uploaded_file.name, 'wb') as f:
        f.write(file_bytes)
    chat.embed(uploaded_file.name)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input(placeholder="Type a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = chat.query_search(prompt)
            response = chat.generate_response(prompt,data)
            placeholder = st.empty()
            full_response = ''
            for text in response:
                full_response += text
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)