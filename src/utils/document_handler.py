import os
from typing import Optional
import uuid
import requests
from config.settings import settings
from fastapi import HTTPException
from .mime_maps import MIME_TYPE_MAP, EXTENSION_MAP

class DocumentHandler:
    def __init__(self, document_path=settings.DOCUMENTS_PATH):
        self.documents_path = document_path
        self._ensure_documents_path()
    
    def _get_doc_type(self, content_type: str, url: str) -> str:
        # Check MIME types first
        for mime, doc_type in MIME_TYPE_MAP.items():
            if mime in content_type.lower():
                return doc_type

        # Check file extension as fallback
        url_lower = url.lower()
        for ext, doc_types in EXTENSION_MAP.items():
            if any(url_lower.endswith(f'.{e}') for e in doc_types):
                return ext

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type. Content-Type: {content_type}, URL: {url}"
        )
    
    def _ensure_documents_path(self):
        if not os.path.exists(self.documents_path):
            os.makedirs(self.documents_path)

    def download_document(self, url: str, filename: Optional[str] = None):
        """
        Download a document from the given URL and save it to the documents path.
        Optimized for faster downloads and processing.
        """
        try:
            # Optimize request with headers and larger timeout
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': '*/*',
                'Connection': 'keep-alive'
            }
            # Increase chunk size for faster downloads
            response = requests.get(
                url,
                stream=True,
                timeout=15,  # Reduced timeout for faster failure detection
                headers=headers
            )
            response.raise_for_status() 
            content_type = response.headers.get('Content-Type', '')
            doc_type = self._get_doc_type(content_type, url)
            if not filename:
                filename = "test"
            file_path = os.path.join(self.documents_path, f"{filename}.{doc_type}")

            # Use larger chunk size for faster file writing
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=32768):  # Doubled from 8192
                    file.write(chunk)

            return file_path, doc_type, filename
        except requests.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download document from {url}. Error: {str(e)}"
            )
    
