"""Service dependencies and initialization module."""

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from services.question_answering import QuestionAnsweringService

# Global service instances that will be initialized at app startup
document_processor = None
vector_store_service = None
qa_service = None

def initialize_services():
    """Initialize all services at application startup."""
    global document_processor, vector_store_service, qa_service
    
    # Initialize services only once
    if document_processor is None:
        document_processor = DocumentProcessor()
    
    if vector_store_service is None:
        vector_store_service = VectorStoreService()
    
    if qa_service is None:
        qa_service = QuestionAnsweringService()
    
    return {
        "document_processor": document_processor,
        "vector_store_service": vector_store_service,
        "qa_service": qa_service
    }

def get_document_processor():
    """Get the document processor service instance."""
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor

def get_vector_store_service():
    """Get the vector store service instance."""
    global vector_store_service
    if vector_store_service is None:
        vector_store_service = VectorStoreService()
    return vector_store_service

def get_qa_service():
    """Get the question answering service instance."""
    global qa_service
    if qa_service is None:
        qa_service = QuestionAnsweringService()
    return qa_service
