from pydantic_settings import BaseSettings

from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE")

SETTINGS = Settings()
