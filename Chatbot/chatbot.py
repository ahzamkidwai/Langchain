from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
 
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
chat_history = [SystemMessage(content="You're a very helpful assistant")]

while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    ans = result.content[0]["text"]
    chat_history.append(AIMessage(content=ans))
    print("AI : ", ans)

print("Chat History : ", chat_history)