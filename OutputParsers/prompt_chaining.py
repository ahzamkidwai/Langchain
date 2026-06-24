from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Groq model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

# First Prompt -> Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Second Prompt -> Summary
template2 = PromptTemplate(
    template="Write a 5-line summary of the following text:\n\n{text}",
    input_variables=["text"]
)

# Generate report
prompt1 = template1.invoke({"topic": "Black Hole"})

print("\nPrompt 1:")
print(prompt1)

result1 = model.invoke(prompt1.to_string())

print("\nDetailed Report:\n")
print(result1.content)

# Generate summary
prompt2 = template2.invoke({"text": result1.content})

result2 = model.invoke(prompt2.to_string())

print("\nSummary:\n")
print(result2.content)