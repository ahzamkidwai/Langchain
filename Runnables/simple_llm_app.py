from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}"
)

topic = input("Enter a topic: ")

formatted_prompt = prompt.format(topic=topic)

response = llm.invoke(formatted_prompt)

print("Generated Blog Title:", response.content)