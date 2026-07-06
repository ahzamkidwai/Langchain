from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
    class Student:
        def __init__(self, name, age, grade):
            self.name = name
            self.age = age
            self.grade = grade
            
        def get_details(self):
            return self.name
            
        def is_passing(self):
            return self.grade >= 6.0
    """

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])