from langchain_core.tools import tool
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.retrieve import retrieve, format_context


@tool
def search_customer_reviews(query: str, score_filter: Optional[int] = None) -> str:
    """
    Search customer reviews using semantic search.

    Use this tool when the user asks about customer opinions, complaints,
    satisfaction, product feedback, delivery experiences, or any question
    that requires understanding what customers have said.

    Do NOT use this tool for churn analysis, revenue questions,
    payment data, or customer spending — use the appropriate
    analytics tools for those instead.

    Args:
        query: A natural language description of what to search for.
               e.g. "late delivery complaints" or "positive product feedback"
        score_filter: Optional integer (1-5). Filter to only reviews with
                      this exact star rating. Use when the user specifically
                      asks about "negative reviews" (1), "1-star reviews" (1),
                      "happy customers" (5), etc. Must be null for general
                      questions — never pass a string, always an integer or null.

    Returns:
        A formatted string containing the most relevant customer reviews
        with their ratings.
    """
    # Defensive type conversion — LLMs sometimes pass "1" instead of 1
    if isinstance(score_filter, str):
        try:
            score_filter = int(score_filter)
        except (ValueError, TypeError):
            score_filter = None

    results = retrieve(query=query, top_k=5, score_filter=score_filter)

    if not results:
        return "No relevant reviews found for this query."

    return format_context(results)