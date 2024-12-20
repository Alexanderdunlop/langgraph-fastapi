from typing import List

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: List[str]
    """The core memories associated with the user."""
    recall_memories: List[str]
    """The recall memories retrieved for the current context."""
