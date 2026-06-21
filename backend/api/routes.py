from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from evals.metrics import run_retrieval_evals
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.graph import app as agent_app
from rag.retrieve import retrieve
from api.schemas import (
    QuestionRequest,
    AnswerResponse,
    ReviewResult,
    IngestResponse,
    HealthResponse
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        message="NeurOps API is running"
    )


@router.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        result = agent_app.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config
        )

        final_message = result["messages"][-1]
        answer = final_message.content

        sources = retrieve(
            query=request.question,
            top_k=request.top_k,
            score_filter=request.score_filter
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

    return AnswerResponse(
        answer=answer,
        sources=[ReviewResult(**r) for r in sources],
        question=request.question
    )


@router.post("/ingest", response_model=IngestResponse)
def trigger_ingest():
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


@router.get("/evals")
def run_evals():
    """
    Runs the NeurOps evaluation suite.
    Tests retrieval accuracy, latency, and similarity scores.
    """
    try:
        results = run_retrieval_evals()
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Eval error: {str(e)}"
        )