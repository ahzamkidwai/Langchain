import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# Load environment variables
load_dotenv()


# -------------------------------
# Configuration
# -------------------------------

VIDEO_ID = "Gfr50f6ZBvo"

QUESTION = (
    "Is the topic of aliens discussed in this video? "
    "If yes, then what was discussed?"
)


# -------------------------------
# Step 1: Get YouTube Transcript
# -------------------------------

def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript from YouTube video.
    """

    try:
        youtube_api = YouTubeTranscriptApi()

        transcript_data = youtube_api.fetch(
            video_id,
            languages=["en"]
        )

        transcript = " ".join(
            chunk.text for chunk in transcript_data
        )

        return transcript

    except TranscriptsDisabled:
        raise Exception(
            "Transcript is disabled for this video."
        )

    except Exception as e:
        raise Exception(
            f"Failed to fetch transcript: {e}"
        )


# -------------------------------
# Step 2: Split Transcript
# -------------------------------

def split_text(transcript: str):
    """
    Split transcript into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    return chunks


# -------------------------------
# Step 3: Create Vector Store
# -------------------------------

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# -------------------------------
# Step 4: Retriever
# -------------------------------

def create_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever


# -------------------------------
# Step 5: Gemini Model
# -------------------------------

def load_model():

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7
    )

    return model


# -------------------------------
# Step 6: RAG Answer Generation
# -------------------------------

def generate_answer(
        question,
        retriever,
        model
):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer only using the provided transcript context.

If the context is insufficient, reply:
"I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=[
            "context",
            "question"
        ]
    )


    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )


    response = model.invoke(
        final_prompt
    )

    return response.content


# -------------------------------
# Main Pipeline
# -------------------------------

def main():

    print("Fetching transcript...")

    transcript = fetch_transcript(
        VIDEO_ID
    )

    print(
        f"Transcript length: {len(transcript)} characters"
    )


    print("Splitting transcript...")

    chunks = split_text(
        transcript
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )


    print("Creating vector database...")

    vector_store = create_vector_store(
        chunks
    )


    retriever = create_retriever(
        vector_store
    )


    print("Loading Gemini model...")

    model = load_model()


    print("\nQuestion:")
    print(QUESTION)


    print("\nGenerating answer...")

    answer = generate_answer(
        QUESTION,
        retriever,
        model
    )


    print("\nAnswer:")
    print(answer)



if __name__ == "__main__":
    main()