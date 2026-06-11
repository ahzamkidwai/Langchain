from langchain_core.prompts import PromptTemplate

# Prompt Template
template = PromptTemplate(
    input_variables=[
        "paper_input",
        "style_input",
        "length_input",
        "additional_prompt"
    ],
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

Additional Instructions:
{additional_prompt}

Requirements:

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain mathematical concepts in simple terms.
   - Provide intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

3. Accuracy:
   - If certain information is not available, respond with:
     "Insufficient information available"
     instead of guessing.

Ensure the summary is clear, accurate, and aligned with the requested style and length.
"""
)

template.save('template.json')