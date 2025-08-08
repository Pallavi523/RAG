"""Document processing service for extracting text and creating document chunks."""

from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.document_handler import DocumentHandler


class DocumentProcessor:
    """Service for processing documents and extracting text."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.document_handler = DocumentHandler()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len,
        )
    
    async def process_documents(self, url: str, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a document from a URL.
        
        Args:
            url: URL of the document to process
            doc_type: Optional document type (pdf, docx, email)
            
        Returns:
            List of document chunks with text and metadata
        """
        # Download the document
        file_path, doc_type, filename = self.document_handler.download_document(url)
        
        # Extract text based on document type
        extension = filename.split('.')[-1].lower() if '.' in filename else ''
        
        if doc_type == 'pdf':
            return await self._process_pdf(file_path, filename)
        elif doc_type in ['docx', 'doc']:
            return await self._process_docx(file_path, filename)
        elif doc_type in ['eml', 'msg']:
            return await self._process_email(file_path, filename)
        else:
            raise ValueError(f"Unsupported document type: {extension}")
    
    async def _process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a PDF document with optimized performance.
        
        Args:
            file_path: Path to the PDF file
            filename: Name of the file
            
        Returns:
            List of document chunks with text and metadata
        """
        # Use PyPDFLoader with optimized settings
        loader = PyPDFLoader(
            file_path,
            extract_images=False  # Skip image extraction for faster processing
        )
        
        # Load documents
        documents = loader.load()
        
        # Add metadata efficiently in a single pass
        metadata = {"source": filename, "file_path": file_path}
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Convert to dictionaries using list comprehension for better performance
        return [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    
    async def _process_docx(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a DOCX document with optimized performance.
        
        Args:
            file_path: Path to the DOCX file
            filename: Name of the file
            
        Returns:
            List of document chunks with text and metadata
        """
        # Use Docx2txtLoader for faster text extraction
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        # Add metadata efficiently in a single pass
        metadata = {"source": filename, "file_path": file_path}
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Convert to dictionaries using list comprehension for better performance
        return [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    
    async def _process_email(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process an email document.
        
        Args:
            file_path: Path to the email file
            filename: Name of the file
            
        Returns:
            List of document chunks with text and metadata
        """
        loader = UnstructuredEmailLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_path"] = file_path
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Convert to dictionaries for easier serialization
        return [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]