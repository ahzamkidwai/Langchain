from langchain_community.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({'a': 5, 'b': 3})
print("Result : ", result);

print("\n\n")

print('result.name : ',multiply.name)
print('result.description : ',multiply.description)
print('result.args : ',multiply.args)