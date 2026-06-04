import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.title("Gemini Chat App")

prompt = st.text_input(label="Enter your prompt", placeholder="Ask anything...")

if st.button("Generate Response"):
    if prompt.strip():
        try:
            with st.spinner("Generating response..."):
                result = model.invoke(prompt)

            st.subheader("Response")

            if isinstance(result.content, str):
                st.write(result.content)

            elif isinstance(result.content, list):
                st.write(result.content[0].get("text", ""))

            else:
                st.write(result.content)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt.") 