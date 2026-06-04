from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

result = embedding.embed_query("Delhi is the capital of India")
print("Embedding length:", len(result))
print("Result in embedding format : ", result)
print(type(result))
print(type(str(result)))