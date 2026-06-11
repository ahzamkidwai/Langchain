from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
 
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
chat_history = []

while True:
    user_input = input("You : ")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    ans = result.content[0]["text"]
    chat_history.append(ans)
    print("AI : ", ans)

print("Chat History : ", chat_history)