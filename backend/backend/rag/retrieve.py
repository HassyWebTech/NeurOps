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
    """
    Returns a Qdrant client pointing to our local storage.
    Same path as ingest.py — they share the same database.
    """
    return QdrantClient(path=QDRANT_LOCAL_PATH)


def get_embedding_model() -> SentenceTransformer:
    """
    Loads the same embedding model used during ingestion.
    On first call it loads from cache — no re-download needed.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve(
    query: str,
    top_k: int = 5,
    score_filter: Optional[int] = None
) -> List[Dict]:
    """
    Takes a natural language question and returns the most
    relevant reviews from Qdrant.

    Parameters:
    - query: the user's question in plain English
    - top_k: how many results to return (default 5)
    - score_filter: optionally filter by review score (1-5)
                    e.g. score_filter=1 returns only 1-star reviews

    Returns a list of dicts, each containing:
    - text: the review content
    - score: the star rating
    - similarity: how closely it matched the query (0-1)
    - order_id: the associated order
    """
    # Step 1: Connect to Qdrant
    client = get_qdrant_client()

    # Step 2: Load embedding model
    model = get_embedding_model()

    # Step 3: Convert the user's question into a vector
    # This is the same process we used for reviews during ingestion
    query_vector = model.encode(query).tolist()

    # Step 4: Build optional filter
    # If the user wants only negative reviews (score=1),
    # we can filter before searching — faster than filtering after
    search_filter = None
    if score_filter is not None:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="score",
                    match=MatchValue(value=score_filter)
                )
            ]
        )

    # Step 5: Search Qdrant
    # Qdrant compares query_vector against all 37,970 stored vectors
    # and returns the top_k most similar ones
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True  # Include the original text and metadata
    )

    # Step 6: Format results into clean dicts
    retrieved = []
    for result in results:
        retrieved.append({
            "text": result.payload.get("text", ""),
            "score": result.payload.get("score", 0),
            "order_id": result.payload.get("order_id", ""),
            "similarity": round(result.score, 4)
            # result.score is the cosine similarity (0-1)
            # Higher = more relevant to the query
        })

    return retrieved


def format_context(results: List[Dict]) -> str:
    """
    Formats retrieved reviews into a clean context string
    that we'll pass to the LLM.

    Instead of dumping raw dicts at the LLM, we format them
    as readable text — this produces better answers.
    """
    if not results:
        return "No relevant reviews found."

    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"Review {i} (Rating: {r['score']}/5):\n{r['text']}"
        )

    return "\n\n".join(context_parts)


if __name__ == "__main__":
    # Quick test — run this file directly to verify retrieval works
    test_query = "late delivery and damaged product"
    print(f"Query: {test_query}\n")

    results = retrieve(test_query, top_k=3)
    print(format_context(results))
    print(f"\nReturned {len(results)} results")
    print(f"Similarity scores: {[r['similarity'] for r in results]}")