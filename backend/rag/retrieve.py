from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    QDRANT_LOCAL_PATH, QDRANT_COLLECTION,
    EMBEDDING_MODEL
)

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_LOCAL_PATH)

def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)

def retrieve(query: str, top_k: int = 5, score_filter: Optional[int] = None) -> List[Dict]:
    client = get_qdrant_client()
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    search_filter = None
    if score_filter is not None:
        search_filter = Filter(
            must=[FieldCondition(key="score", match=MatchValue(value=score_filter))]
        )
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True
    ).points
    retrieved = []
    for result in results:
        retrieved.append({
            "text": result.payload.get("text", ""),
            "score": result.payload.get("score", 0),
            "order_id": result.payload.get("order_id", ""),
            "similarity": round(result.score, 4)
        })
    return retrieved

def format_context(results: List[Dict]) -> str:
    if not results:
        return "No relevant reviews found."
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"Review {i} (Rating: {r['score']}/5):\n{r['text']}")
    return "\n\n".join(context_parts)
