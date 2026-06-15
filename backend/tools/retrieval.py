from langchain_core.tools import tool
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.retrieve import retrieve, format_context


@tool
def search_customer_reviews(query: str, score_filter: int = None) -> str:
    """
    Search customer reviews using semantic search.

    Use this tool when the user asks about customer opinions, complaints,
    satisfaction, product feedback, delivery experiences, or any question
    that requires understanding what customers have said.

    Args:
        query: A natural language description of what to search for.
               e.g. "late delivery complaints" or "positive product feedback"
        score_filter: Optional. Filter to only reviews with this exact
                       star rating (1-5). Use this when the user specifically
                       asks about "negative reviews", "1-star reviews",
                       "happy customers" (5), etc. Leave as None for
                       general questions.

    Returns:
        A formatted string containing the most relevant customer reviews
        with their ratings.
    """
    results = retrieve(query=query, top_k=5, score_filter=score_filter)

    if not results:
        return "No relevant reviews found for this query."

    return format_context(results)