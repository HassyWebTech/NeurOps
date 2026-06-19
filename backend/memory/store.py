import json
import os

# All business context is stored in this single JSON file.
# In production, this would become a PostgreSQL table keyed
# by business_id — same functions, different storage backend.
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "business_context.json")


def load_context() -> dict:
    """
    Loads saved business context from disk.

    Returns an empty dict if no memory file exists yet —
    this is the first-run case, completely normal.
    """
    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_context(key: str, value: str) -> dict:
    """
    Saves a single fact to business context and persists to disk.

    Example: save_context("business_type", "fashion retail in Lagos")

    Returns the full updated context dict.
    """
    context = load_context()
    context[key] = value

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    return context


def format_context_for_prompt(context: dict) -> str:
    """
    Formats saved business context as readable text for
    injection into the agent's system prompt.

    Returns an empty string if there's no context yet —
    so the prompt stays clean when nothing has been saved.
    """
    if not context:
        return ""

    lines = ["Known business context:"]
    for key, value in context.items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)