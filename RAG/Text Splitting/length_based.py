from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('survey.pdf')

docs = loader.load()

text="""A text splitter is a tool or process that breaks a large piece of text into smaller, manageable sections. It is commonly used in natural language processing (NLP), data preprocessing, and document management. By dividing long documents into smaller chunks, text splitters make it easier for computers to process, analyze, and retrieve information efficiently.

Text splitters can divide text in different ways depending on the purpose. Some split text by sentences, paragraphs, or fixed numbers of characters, while others use tokens or semantic meaning to create meaningful chunks. Choosing the right splitting method helps preserve context and improves the accuracy of tasks such as text summarization, question answering, and information retrieval.

Understanding text splitters is especially important when working with large language models (LLMs) and AI applications. Since many AI models have limits on the amount of text they can process at one time, splitting documents into well-structured chunks ensures that important information is retained and processed effectively. A good text-splitting strategy leads to better performance, faster processing, and more accurate results in AI-powered systems.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result1 = splitter.split_text(text)

result2 = splitter.split_documents(docs)

print ("Result : \n\n\n", result2[0])
print("\n\n\n\nLEN : ", len(result2))
