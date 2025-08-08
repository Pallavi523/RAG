from fastapi import APIRouter, HTTPException, status, Depends
from schema.hackrx import HackRxRunRequest, HackRxRunResponse
from services.dependencies import get_document_processor, get_vector_store_service, get_qa_service
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from services.question_answering import QuestionAnsweringService
router = APIRouter(
    prefix="/hackrx",
    tags=["hackrx"],

)
@router.post("/run", response_model=HackRxRunResponse)
async def question_answering(
    request: HackRxRunRequest,
    document_processor: DocumentProcessor = Depends(get_document_processor),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    qa_service: QuestionAnsweringService = Depends(get_qa_service)
):
    """
    Run a HackRx task with the provided documents and questions.
    Uses pre-initialized service instances for better performance.
    """
    try:
        documents = []
        
        # Handle both single URL and list of URLs
        urls = [request.documents] if not isinstance(request.documents, list) else request.documents
  
        for url in urls:
            url_str = str(url)

            if not url_str.startswith("http://") and not url_str.startswith("https://"):
                print(f"Skipping invalid URL: {url_str}")
                continue

            doc_chunks = await document_processor.process_documents(url_str)
            documents.extend(doc_chunks)

        if not documents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents processed successfully.")
        print(f"Processed {len(documents)} documents successfully.")

        vector_store = await vector_store_service.create_vector_store(documents)

        await vector_store_service.save_vector_store(vector_store, "hackrx_index")
        detailed_answers = await qa_service.batch_answer_questions(vector_store, request.questions)
        
        simple_response = HackRxRunResponse(
            answers=[answer["answer"] for answer in detailed_answers]
        )
        
        return simple_response
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    