from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, GROQ_MODEL
from agents.state import AgentState
from tools.retrieval import search_customer_reviews


# All tools the agent has access to.
# Adding a new tool here (Week 4-5) is all that's needed —
# the agent automatically learns to use it via its description.
TOOLS = [search_customer_reviews]

# The LLM that powers the agent's reasoning.
# bind_tools() tells the LLM what tools exist and lets it
# decide when to call them.
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3
).bind_tools(TOOLS)


SYSTEM_PROMPT = """You are NeurOps, an intelligent business analyst AI.

You help business owners understand their customers by analyzing reviews
and data. You have access to tools that search customer reviews.

When answering:
- Use tools to gather real evidence before answering
- Be direct and specific, referencing patterns across reviews
- Highlight both problems and positives
- Suggest actionable improvements when relevant
- Keep answers concise but comprehensive"""


def agent_node(state: AgentState) -> AgentState:
    """
    The 'thinking' node. The LLM looks at the conversation so far
    and decides: call a tool, or respond directly.

    If it decides to call a tool, the returned AIMessage will
    contain a 'tool_calls' field. LangGraph reads this to know
    where to route next.
    """
    messages = state["messages"]

    # Prepend the system prompt on every call so the agent
    # always knows its role and available tools
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(full_messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    The routing function. After the agent node runs, this decides
    where to go next:
    - If the last message has tool_calls -> go to "tools" node
    - Otherwise -> end (the agent gave a final answer)
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


# ── Build the graph ──

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

# The graph always starts at "agent"
graph.set_entry_point("agent")

# After "agent" runs, conditionally route to "tools" or END
graph.add_conditional_edges("agent", should_continue)

# After "tools" runs, always go back to "agent" so it can
# process the tool results and decide what to do next
graph.add_edge("tools", "agent")

# Compile into a runnable application
app = graph.compile()