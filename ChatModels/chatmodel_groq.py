from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

result = model.invoke("Write one small paragraph of Lucknow, UP.")
# print("Result is : ", result)

answer = result.content
print("Answer is : ", answer)