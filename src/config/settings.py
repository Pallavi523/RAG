from pydantic_settings import BaseSettings
import os
class Settings(BaseSettings):
    API_KEY: str
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_API_VERSION: str = "2023-05-15"  
    FAISS_INDEX_PATH: str = "faiss_index"
    DOCUMENTS_PATH: str = "documents"
    DOCUMENT_STORAGE_PATH: str = "document_storage"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()

os.makedirs(settings.DOCUMENTS_PATH, exist_ok=True)
os.makedirs(settings.DOCUMENT_STORAGE_PATH, exist_ok=True)
os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)