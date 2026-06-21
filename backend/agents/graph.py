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
from tools.memory import save_business_context
from tools.churn import churn_analysis
from tools.analytics import customer_analytics
from memory.store import load_context, format_context_for_prompt


TOOLS = [search_customer_reviews, save_business_context,
         churn_analysis, customer_analytics]

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

    # Load any previously saved business context and inject it
    # into the system prompt — this is long-term memory in action.
    # Every run, the agent "remembers" facts from past sessions.
    business_context = load_context()
    context_text = format_context_for_prompt(business_context)

    system_content = SYSTEM_PROMPT
    if context_text:
        system_content += f"\n\n{context_text}"

    full_messages = [SystemMessage(content=system_content)] + list(messages)

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

# MemorySaver stores conversation state in memory, keyed by thread_id.
# Each unique thread_id gets its own conversation history.
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# Compile into a runnable application with memory enabled
app = graph.compile(checkpointer=checkpointer)