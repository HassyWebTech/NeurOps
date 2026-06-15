from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The shared state that flows through every node in our graph.

    - messages: the full conversation history (user question, AI thoughts,
      tool calls, tool results). `add_messages` is a special reducer —
      instead of replacing this list each time a node runs, it appends
      new messages to it. This is how the agent "remembers" what
      happened in earlier steps of the same run.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]