from pydantic import BaseModel
from typing import List, Optional


class QuestionRequest(BaseModel):
    """
    What the frontend sends to the backend when a user asks a question.
    """
    question: str
    top_k: int = 5
    score_filter: Optional[int] = None
    thread_id: str = "default"  # Identifies the conversation for memory

class ReviewResult(BaseModel):
    """
    A single retrieved review returned to the frontend.
    """
    text: str
    score: int
    order_id: str
    similarity: float


class AnswerResponse(BaseModel):
    """
    What the backend sends back to the frontend after processing.
    
    Example response body:
    {
        "answer": "Customers frequently complain about...",
        "sources": [...],
        "question": "What are customers saying about late deliveries?"
    }
    """
    answer: str                      # The LLM generated answer
    sources: List[ReviewResult]      # The reviews used to generate the answer
    question: str                    # Echo back the original question


class IngestResponse(BaseModel):
    """
    Response after triggering data ingestion.
    """
    message: str
    total_ingested: int


class HealthResponse(BaseModel):
    """
    Simple health check response.
    Used to verify the API is running.
    """
    status: str
    message: str