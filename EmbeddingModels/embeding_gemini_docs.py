from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital city of West Bengal",
    "Paris is the cpaital of France",
    "London is the capital of United Kingdom"
]

result = embedding.embed_documents(documents)
print("Embedding length:", len(result))
print("Result in embedding format : ", result)
print(type(result))
print(type(str(result)))