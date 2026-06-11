# import streamlit as st
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate

# load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# st.title("Gemini Chat App")

# paper_input = st.selectbox("Select Research Paper Name", ["Attention is All You Need", "BERT: Pre-Training of Deep Bidirectional Transformers", "GPT-3: language models are Few-Shot Learners", "Diffusion Models Beats GANs on Image Synthesis"])

# style_input = st.selectbox("Select Explaination Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

# length_input = st.selectbox("Select Explaination Length", ["Short (1-2 Paragraphs)", "Medium (3-5 Paragraphs)", "Long (Detailed Explaination)"])

# template = PromptTemplate(
#     template = """Please summarize the research paper titled "{paper_input}" with the following specifications:
#     Explaination Style: {style_input}
#     Explaination Length: {length_input}
#     1. Mathematical Details:
#         - Include relevant mathematical equations if present in the paper.
#         - Explain the mathematical concepts using simple, intutive code snippets where applicable.
#     2. Analogies
#         - Use relatable analogies to simplify complex ideas.
#     If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
#     Ensure the summary is clear, accurate, and aligned with the provided style and length.
#     """
# )

# prompt = st.text_input(label="Enter your prompt", placeholder="Ask anything...")

# if st.button("Generate Response"):
#     if prompt.strip():
#         try:
#             with st.spinner("Generating response..."):
#                 result = model.invoke(prompt)

#             st.subheader("Response")

#             if isinstance(result.content, str):
#                 st.write(result.content)

#             elif isinstance(result.content, list):
#                 st.write(result.content[0].get("text", ""))

#             else:
#                 st.write(result.content)

#         except Exception as e:
#             st.error(f"Error: {e}")
#     else:
#         st.warning("Please enter a prompt.") 






import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt

# Load environment variables
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Streamlit UI
st.title("Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention is All You Need",
        "BERT: Pre-Training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 Paragraphs)",
        "Medium (3-5 Paragraphs)",
        "Long (Detailed Explanation)"
    ]
)

template = load_prompt('template.json')

additional_prompt = st.text_area(
    "Additional Instructions (Optional)",
    placeholder="E.g. Focus on practical applications"
)

# Prompt Template
# template = PromptTemplate(
#     input_variables=[
#         "paper_input",
#         "style_input",
#         "length_input",
#         "additional_prompt"
#     ],
#     template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:

# Explanation Style: {style_input}
# Explanation Length: {length_input}

# Additional Instructions:
# {additional_prompt}

# Requirements:

# 1. Mathematical Details:
#    - Include relevant mathematical equations if present in the paper.
#    - Explain mathematical concepts in simple terms.
#    - Provide intuitive code snippets where applicable.

# 2. Analogies:
#    - Use relatable analogies to simplify complex ideas.

# 3. Accuracy:
#    - If certain information is not available, respond with:
#      "Insufficient information available"
#      instead of guessing.

# Ensure the summary is clear, accurate, and aligned with the requested style and length.
# """
# )

if st.button("Generate Summary"):
    try:
        with st.spinner("Generating summary..."):

            final_prompt = template.format(
                paper_input=paper_input,
                style_input=style_input,
                length_input=length_input,
                additional_prompt=additional_prompt
            )

            response = model.invoke(final_prompt)

        st.subheader("Summary")
        st.write(response.content)

    except Exception as e:
        st.error(f"Error: {str(e)}")