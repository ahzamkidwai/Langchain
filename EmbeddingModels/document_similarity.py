from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

document = [
    "Virat Kohli is one of the greatest batsmen in Indian cricket history.",
    "Sachin Tendulkar is often called the God of Cricket in India.",
    "MS Dhoni led India to victory in the 2007 T20 World Cup and 2011 ODI World Cup.",
    "Rohit Sharma holds the record for the highest individual score in ODI cricket.",
    "The Indian cricket team is one of the strongest teams in international cricket today."
]

query = "Tell me about Sachin Tendulkar"

doc_emb = embeddings.embed_documents(document)
query_emb = embeddings.embed_query(query)

scores = cosine_similarity([query_emb], doc_emb)[0]

print(scores)
print("Best match:", document[np.argmax(scores)])