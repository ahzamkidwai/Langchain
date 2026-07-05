from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

prompt = PromptTemplate(
    template='Write a summary for the following poem - {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('poem.txt', encoding='utf-8')
print("Loader : ", loader)
docs = loader.load()

print("\n\nPage Content : ", docs[0].page_content)
print("\n\nMeta Data : ", docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({ 'poem' : docs[0].page_content })
print("\n\n\nRESULT : ", result)