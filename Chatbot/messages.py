from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

messages = [
    SystemMessage(content="You're a helpful assistant."),
    HumanMessage(content="Tell me about langchain.")
]

result = model.invoke(messages)
ans = result.content[0]["text"]
messages.append(AIMessage(content=ans))

print(messages)