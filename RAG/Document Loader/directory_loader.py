from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='pdfs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(len(docs))

print("\n\nPage Content : ", docs[0].page_content)
print("\n\nMetaData : ", docs[0].metadata)