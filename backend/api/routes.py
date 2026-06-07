from fastapi import APIRouter, HTTPException
from groq import Groq
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, GROQ_MODEL
from rag.retrieve import retrieve, format_context
from api.schemas import (
    QuestionRequest,
    AnswerResponse,
    ReviewResult,
    IngestResponse,
    HealthResponse
)

# APIRouter groups all our endpoints together
# main.py will register this router with the FastAPI app
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple endpoint to verify the API is running.
    No logic — just returns a status message.
    """
    return HealthResponse(
        status="ok",
        message="NeurOps API is running"
    )


@router.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    The core RAG endpoint — the full pipeline in one call.

    Step 1: Retrieve relevant reviews from Qdrant
    Step 2: Format them as context
    Step 3: Send context + question to Groq LLM
    Step 4: Return the answer + source reviews
    """

    # Step 1: Retrieve relevant reviews
    # We ask for more than top_k so the LLM has richer context
    results = retrieve(
        query=request.question,
        top_k=request.top_k,
        score_filter=request.score_filter
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant reviews found for this question"
        )

    # Step 2: Format retrieved reviews as readable context
    context = format_context(results)

    # Step 3: Build the prompt for the LLM
    # This is prompt engineering — how we structure the input
    # determines the quality of the output
    system_prompt = """You are NeurOps, an intelligent business analyst AI.
You analyze customer reviews and provide clear, actionable insights to business owners.

When answering:
- Be direct and specific
- Reference patterns across multiple reviews
- Highlight both problems and positives
- Suggest actionable improvements when relevant
- Keep answers concise but comprehensive"""

    user_prompt = f"""Based on the following customer reviews, answer this question:

Question: {request.question}

Customer Reviews:
{context}

Provide a clear, insightful answer based on the reviews above."""

    # Step 4: Call Groq LLM
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower = more factual, less creative
            max_tokens=1024
        )
        answer = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM error: {str(e)}"
        )

    # Step 5: Return answer + sources
    return AnswerResponse(
        answer=answer,
        sources=[ReviewResult(**r) for r in results],
        question=request.question
    )


@router.post("/ingest", response_model=IngestResponse)
def trigger_ingest():
    """
    Triggers the ingestion pipeline programmatically.
    Useful for re-ingesting after data updates.
    """
    try:
        from rag.ingest import ingest
        ingest()
        return IngestResponse(
            message="Ingestion complete",
            total_ingested=0
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion error: {str(e)}"
        )