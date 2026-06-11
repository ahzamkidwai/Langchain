from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
 
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
# model = ChatGroq(model="openai/gpt-oss-20b")


while True:
    user_input = input("You : ")
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    ans = result.content[0]["text"]
    print("AI : ", ans)
