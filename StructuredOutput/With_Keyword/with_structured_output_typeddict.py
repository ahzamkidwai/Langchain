from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(
    # model="gemini-2.5-flash"
    model="gemini-3.5-flash"
)

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral/mixed"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """The Samsung Galaxy S26 Ultra is Samsung’s latest flagship smartphone, designed for users who want top-tier performance, premium design, and advanced camera capabilities. It features a large 6.9-inch Dynamic AMOLED 2X display with QHD+ resolution and a 120Hz adaptive refresh rate, delivering vibrant colors, sharp visuals, and smooth scrolling. The device also has a slimmer and lighter design compared to previous Ultra models, making it more comfortable to use despite its large screen size.
    Performance is one of the strongest aspects of the Galaxy S26 Ultra. Powered by the Snapdragon 8 Elite Gen 5 processor, the phone offers exceptional speed for gaming, multitasking, and AI-powered features. It is available with up to 16GB of RAM and as much as 1TB of storage, ensuring ample space and smooth performance for demanding users. Samsung has also integrated advanced Galaxy AI features to improve productivity, photography, and everyday smartphone interactions .
    The camera system is another major highlight of the S26 Ultra. It includes a 200MP main camera, a 50MP ultra-wide camera, a 50MP periscope telephoto camera with 5x optical zoom, and an additional 10MP telephoto lens with 3x optical zoom. This versatile setup allows users to capture detailed photos, stunning landscapes, and high-quality zoom shots. Enhanced image processing and AI features further improve photography performance in various lighting conditions.
    Battery life remains impressive thanks to a 5,000mAh battery that supports faster charging than previous generations. The phone combines premium hardware, advanced cameras, AI-powered software, and a refined design, making it one of the most powerful Android smartphones available in 2026. The Galaxy S26 Ultra continues Samsung’s tradition of delivering a feature-rich flagship experience for professionals, creators, and technology enthusiasts."""
)

print(result)
print(result['summary'])
print(result['sentiment'])