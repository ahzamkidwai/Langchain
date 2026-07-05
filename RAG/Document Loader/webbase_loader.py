from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/realme-gt-7t-icesense-black-256-gb/p/itmda26d662ee28e'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print("\n\n\n", docs[0])