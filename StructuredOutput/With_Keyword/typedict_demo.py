from typing import TypedDict

class Person(TypedDict):    
    name: str
    age: str

new_person: Person = {'name': 'Somanshu Sharma', 'age': 24}
print(new_person)