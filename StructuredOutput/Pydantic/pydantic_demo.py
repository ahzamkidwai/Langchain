from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    first_name: str
    middle_name: Optional[str] = '' # Optional Fields
    sir_name: str = 'kidwai' # Adding Default Value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=7) # Add a float range of cgpi grade from 0 to 10 (Add a default cgpa value of 7)
    
    
new_student = {'first_name': 'ahzam', 'age': '24', 'email': 'ahzam@gmail.com'} # Automatically Type casting from string to integer

print('new_student : ', new_student)

student = Student(**new_student)
print('student : ', student)

student_dict = dict(student) # Converting the object into dictonary
print('student_dict [age] : ', student_dict['age'])
print('student_dict [sir_name] : ', student_dict['sir_name'])

student_json = student.model_dump_json()
print('student_json : ', student_json)