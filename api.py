"""
FastAPI REST API for the DSPy RAG System.

Provides endpoints for:
- Question answering with RAG
- Document management
- System health checks
"""
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import dspy

from config import settings
from vector_store import VectorStore, create_retriever
from knowledge_base import get_sample_documents
from dspy_modules import BasicRAG, AdvancedRAG, MultiHopQA, DocumentSummarizer


# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


# ============================================================
# Global State
# ============================================================

class AppState:
    """Application state container."""
    vector_store: Optional[VectorStore] = None
    basic_rag: Optional[BasicRAG] = None
    advanced_rag: Optional[AdvancedRAG] = None
    multihop_qa: Optional[MultiHopQA] = None
    summarizer: Optional[DocumentSummarizer] = None


app_state = AppState()


# ============================================================
# Pydantic Models for API
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    mode: str = Field(
        default="basic",
        description="RAG mode: 'basic', 'advanced', or 'multihop'"
    )


class QuestionResponse(BaseModel):
    question: str
    answer: str
    context: str
    mode: str
    additional_info: Optional[dict] = None


class DocumentRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document texts to add")
    ids: Optional[List[str]] = Field(None, description="Optional document IDs")


class DocumentResponse(BaseModel):
    message: str
    count: int


class SummarizeRequest(BaseModel):
    document: str = Field(..., description="Document text to summarize")


class SummarizeResponse(BaseModel):
    summary: str


class HealthResponse(BaseModel):
    status: str
    model: str
    document_count: int


# ============================================================
# Security
# ============================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Verify API key if configured."""
    if not settings.APP_API_KEY:
        # Authentication disabled
        return None

    if api_key != settings.APP_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key


# ============================================================
# Lifespan Management
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    logger.info("Initializing DSPy application...")

    # Configure DSPy with Google Gemini
    lm = dspy.LM(
        model=f"google/{settings.GOOGLE_MODEL}",
        api_key=settings.GOOGLE_API_KEY
    )
    dspy.configure(lm=lm)
    logger.info(f"Configured DSPy with model: {settings.GOOGLE_MODEL}")

    # Initialize vector store
    app_state.vector_store = VectorStore(
        collection_name="knowledge_base",
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    # Load sample documents if empty
    if app_state.vector_store.count() == 0:
        logger.info("Loading sample documents...")
        docs = get_sample_documents()
        app_state.vector_store.add_documents(
            documents=[d["text"] for d in docs],
            ids=[d["id"] for d in docs],
            metadatas=[d["metadata"] for d in docs]
        )
        logger.info(f"Loaded {len(docs)} sample documents")

    # Create retriever
    retriever = create_retriever(app_state.vector_store, k=3)

    # Initialize DSPy modules
    app_state.basic_rag = BasicRAG(retriever)
    app_state.advanced_rag = AdvancedRAG(retriever, max_hops=2)
    app_state.multihop_qa = MultiHopQA(retriever, max_hops=3)
    app_state.summarizer = DocumentSummarizer()

    logger.info("Application initialized successfully!")

    yield

    # Cleanup
    logger.info("Shutting down application...")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="DSPy RAG Demo API",
    description="A complete RAG system using DSPy with Google Gemini",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health status."""
    return HealthResponse(
        status="healthy",
        model=settings.GOOGLE_MODEL,
        document_count=app_state.vector_store.count() if app_state.vector_store else 0
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    _: Optional[str] = Depends(verify_api_key)
):
    """
    Answer a question using RAG.

    Modes:
    - basic: Simple retrieval and generation
    - advanced: Query rewriting with answer assessment
    - multihop: Iterative retrieval for complex questions
    """
    logger.info(f"Received question: {request.question} (mode: {request.mode})")

    try:
        if request.mode == "basic":
            result = app_state.basic_rag(question=request.question)
            additional_info = None

        elif request.mode == "advanced":
            result = app_state.advanced_rag(question=request.question)
            additional_info = {
                "search_query": result.search_query,
                "assessment": result.assessment,
                "reasoning": result.reasoning
            }

        elif request.mode == "multihop":
            result = app_state.multihop_qa(question=request.question)
            additional_info = {
                "queries_used": result.queries
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Use 'basic', 'advanced', or 'multihop'"
            )

        return QuestionResponse(
            question=request.question,
            answer=result.answer,
            context=result.context,
            mode=request.mode,
            additional_info=additional_info
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents", response_model=DocumentResponse)
async def add_documents(
    request: DocumentRequest,
    _: Optional[str] = Depends(verify_api_key)
):
    """Add new documents to the knowledge base."""
    try:
        app_state.vector_store.add_documents(
            documents=request.documents,
            ids=request.ids
        )

        return DocumentResponse(
            message="Documents added successfully",
            count=len(request.documents)
        )

    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(
    request: SummarizeRequest,
    _: Optional[str] = Depends(verify_api_key)
):
    """Summarize a document."""
    try:
        result = app_state.summarizer(document=request.document)
        return SummarizeResponse(summary=result.summary)

    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/count")
async def get_document_count():
    """Get the number of documents in the knowledge base."""
    return {"count": app_state.vector_store.count() if app_state.vector_store else 0}


@app.delete("/documents")
async def clear_documents(_: Optional[str] = Depends(verify_api_key)):
    """Clear all documents from the knowledge base."""
    try:
        app_state.vector_store.clear()
        return {"message": "All documents cleared"}

    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(
    query: str,
    n_results: int = 3,
    _: Optional[str] = Depends(verify_api_key)
):
    """Search for relevant documents."""
    try:
        results = app_state.vector_store.search_with_metadata(
            query=query,
            n_results=n_results
        )
        return results

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
