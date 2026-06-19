from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, ToolMessage
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
    """
    Simple endpoint to verify the API is running.
    """
    return HealthResponse(
        status="ok",
        message="NeurOps API is running"
    )


@router.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    The agent endpoint. The question is sent into our LangGraph agent,
    which decides whether to search reviews, calls the tool if needed,
    and produces a final answer.

    For source citations, we separately call retrieve() with the
    original question — this gives the frontend reviews to display
    even if the agent's tool query was phrased differently.
    """
    try:
       # Step 1: Run the agent
        # config tells the checkpointer which conversation thread
        # to load history from and save updates to
        config = {"configurable": {"thread_id": request.thread_id}}

        result = agent_app.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config
        )

        # Step 2: Extract the final answer (last message in the conversation)
        final_message = result["messages"][-1]
        answer = final_message.content

        # Step 3: Get source reviews for citation
        # We use the original question + user's filter directly,
        # independent of what query the agent's tool used internally.
        # This guarantees sources are always shown to the user.
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
    """
    Triggers the ingestion pipeline programmatically.
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