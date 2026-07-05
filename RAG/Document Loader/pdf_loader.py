from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('survey.pdf')

docs = loader.load()
print("Docs : \n\n", docs[0].page_content)
print("\n\n\n\nMetaData : \n\n", docs[1].metadata)