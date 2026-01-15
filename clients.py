from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatPerplexity
import os

load_dotenv()


def get_gemini_client(config={}):
    api_key_gemini = os.getenv("GEMINI_API_KEY")
    if not api_key_gemini:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    if not config:
        config = {"model": "gemini-2.5-flash-lite", "temperature": 0}

    client = ChatGoogleGenerativeAI(**config, api_key=api_key_gemini)
    return client


def get_gemini_embeddings():
    api_key_gemini = os.getenv("GEMINI_API_KEY")
    if not api_key_gemini:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", api_key=api_key_gemini
    )
    return embedding


def get_gemini_chat(config={}):
    api_key_gemini = os.getenv("GEMINI_API_KEY")
    if not api_key_gemini:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    if not config:
        config = {"model": "gemini-3-flash-preview", "temperature": 0}

    client = ChatGoogleGenerativeAI(**config, api_key=api_key_gemini)
    return client


def get_chroma_vectorstore(embeddings=None):
    db = Chroma(embedding_function=embeddings, persist_directory="./db")
    return db


def get_groq_chat(config={}):
    api_key_groq = os.getenv("GROQ_API_KEY")
    if not api_key_groq:
        raise ValueError("GROQ_API_KEY not found in .env file")

    if not config:
        config = {"model": "openai/gpt-oss-120b", "temperature": 0}

    # IMPORTANT: LangChain's ChatGroq uses specific param names
    client = ChatGroq(
        model_name=config["model"],
        api_key=api_key_groq,
        temperature=config["temperature"],
    )
    return client


def get_perplexity_chat(config={}):
    api_key_pplx = os.getenv("PPLX_API_KEY")
    if not api_key_pplx:
        raise ValueError("PPLX_API_KEY not found in .env file")

    if not config:
        config = {"model": "sonar", "temperature": 0}

    client = ChatPerplexity(
        model=config["model"],
        pplx_api_key=api_key_pplx,
        temperature=config["temperature"],
    )
    return client
