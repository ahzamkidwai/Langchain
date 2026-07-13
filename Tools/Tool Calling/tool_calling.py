from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import requests

load_dotenv()

# Create Tool
@tool 
def multiply (a: int, b: int) -> int:
    """Given 2 numbers a and b, this tool returns their product"""
    return a*b

print(multiply.invoke({'a': 3, 'b': 4}))

print("Tool Name : ", multiply.name)
print("Tool Description : ", multiply.description)
print("Tool Arguments : ", multiply.args)


# Tool Binding 
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.7
)

llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage("Can you multiply 3 and 12?")
messages = [query]

# First LLM call
result = llm_with_tools.invoke(messages)
messages.append(result)

# Execute tool
tool_args = result.tool_calls[0]["args"]
tool_output = multiply.invoke(tool_args)

messages.append(
    ToolMessage(
        content=str(tool_output),
        tool_call_id=result.tool_calls[0]["id"],
    )
)

# Final LLM response
final_response = llm_with_tools.invoke(messages)

print(final_response.content)