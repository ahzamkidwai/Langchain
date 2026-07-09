from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

sample = """
Artificial intelligence is transforming industries by automating repetitive tasks, improving decision-making, and enabling smarter applications such as virtual assistants and recommendation systems. Businesses use AI to analyze large amounts of data and identify patterns that humans might miss. On the other hand, maintaining a healthy diet is essential for overall well-being. Eating a balanced mix of fruits, vegetables, whole grains, and proteins helps strengthen the immune system, maintain energy levels, and reduce the risk of chronic diseases.

Space exploration has expanded our understanding of the universe through satellites, space telescopes, and missions to planets such as Mars. Scientists continue to search for evidence of past life and develop technologies for future human missions beyond Earth. Meanwhile, cricket remains one of the most popular sports in many countries, attracting millions of fans worldwide. The game requires teamwork, strategic planning, and individual skill, with international tournaments bringing together the best players from different nations.
"""

docs = text_splitter.create_documents([sample])

print(f"Number of chunks: {len(docs)}\n")

for i, doc in enumerate(docs, start=1):
    print(f"------ Chunk {i} ------")
    print(doc.page_content)
    print()