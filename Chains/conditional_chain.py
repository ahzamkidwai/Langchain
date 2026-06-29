from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
)

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

str_parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

classifier_prompt = PromptTemplate(
    template="""
Classify the following feedback as either positive or negative.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    },
)

positive_prompt = PromptTemplate(
    template="""
Write a polite thank-you response to this positive feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="""
Write a polite apology and support response to this negative feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

classifier_chain = classifier_prompt | model | pydantic_parser

parallel_chain = RunnableParallel(
    feedback=RunnableLambda(lambda x: x["feedback"]),
    sentiment=classifier_chain,
)

branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        positive_prompt | model | str_parser,
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        negative_prompt | model | str_parser,
    ),
    RunnableLambda(lambda x: "Could not determine sentiment."),
)

chain = parallel_chain | branch_chain

result = chain.invoke(
    {
        "feedback": "This is a terrible phone."
    }
)

print(result)