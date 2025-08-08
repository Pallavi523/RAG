"""Question answering service using LangChain and Gemini LLM."""

from typing import List, Dict, Any, Optional

from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import BaseRetriever
from config.settings import settings

# Pull the retrieval QA prompt from LangChain Hub
retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")


class QuestionAnsweringService:
    """Service for answering questions based on document context."""

    def __init__(self):
        """Initialize the question answering service."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Using Flash for higher rate limits
            temperature=0.2,
            max_tokens=100,
            google_api_key=settings.GOOGLE_API_KEY,
            # Add retry and timeout settings for better rate limit handling
            max_retries=3,
            request_timeout=60,
        )

    async def answer_question(
        self, vector_store: FAISS, question: str
    ) -> Dict[str, Any]:
        """Answer a question based on the document context.

        Args:
            vector_store: FAISS vector store containing document embeddings
            question: Question to answer

        Returns:
            Dictionary with answer and metadata
        """
        combine_docs_chain = create_stuff_documents_chain(self.llm, retrieval_qa_prompt)

        retriever = vector_store.as_retriever(
            search_type="mmr",  
            search_kwargs={
                "k": 10,  
                "fetch_k": 10,  
                "lambda_mult": 0.5,  
            }
        )
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        result = rag_chain.invoke({"input": question})

        source_docs = result.get("context", [])

        return {
            "question": question,
            "answer": result["answer"],
            "context": [doc.page_content for doc in source_docs],
            "sources": [doc.metadata.get("source", "unknown") for doc in source_docs],
        }

    async def batch_answer_questions(
        self, vector_store: FAISS, questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions based on the document context.

        Args:
            vector_store: FAISS vector store containing document embeddings
            questions: List of questions to answer

        Returns:
            List of dictionaries with answers and metadata
        """
        results = []
        for question in questions:
            answer = await self.answer_question(vector_store, question)
            results.append(answer)

        return results