from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")
    
def multiply_function(a: int, b: int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply_function,
    name="Multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 10, 'b': 20})

print("Result using custom tool : ", result)