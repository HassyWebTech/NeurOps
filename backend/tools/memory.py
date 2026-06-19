from langchain_core.tools import tool
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from memory.store import save_context


@tool
def save_business_context(key: str, value: str) -> str:
    """
    Save a fact about the business for future conversations.

    Use this tool when the user shares information about their business
    that would be useful to remember later — such as business type,
    location, target market, products sold, or business goals.

    Examples of when to use this:
    - "I run a fashion store in Lagos" -> key="business_type", value="fashion retail in Lagos"
    - "We mainly target young professionals" -> key="target_market", value="young professionals"
    - "My business is called Aura Boutique" -> key="business_name", value="Aura Boutique"

    Args:
        key: A short identifier for this fact, in snake_case.
             e.g. "business_type", "location", "target_market", "business_name"
        value: The actual information to remember, as a clear sentence or phrase.

    Returns:
        A confirmation message.
    """
    save_context(key, value)
    return f"Saved: {key} = {value}"