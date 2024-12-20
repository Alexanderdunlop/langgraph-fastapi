from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set")

class AgentState(TypedDict):
    # messages: Annotated[List[AnyMessage], add_messages]
    messages: List[Union[HumanMessage, AIMessage]]
    """The messages in the conversation."""
    current_step: int
    """The current step in the conversation."""
    max_steps: int
    """The maximum number of steps in the conversation."""
    # core_memories: List[str]
    # """The core memories associated with the user."""
    # recall_memories: List[str]
    # """The recall memories retrieved for the current context."""

# def should_continue(state):
#     """Determine whether to use tools or end the conversation based on the last message.
    
#     Args:
#         state (schemas.State): The current state of the conversation.

#     Returns:
#         str: "end" if the conversation should end, "continue" if tools should be used.
#     """
#     messages = state["messages"]
#     last_message = messages[-1]
#     if not last_message.tool_calls:
#         return "end"
#     else:
#         return "continue"
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

def agent(state: AgentState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    messages.append(response)
    state["current_step"] += 1
    
    if state["current_step"] >= state["max_steps"]:
        return {"state": state, "next": END}
    return {"state": state, "next": "agent"}

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
# workflow.add_node("agent", call_model)
# workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
# workflow.add_edge("load_memories", "agent")
workflow.add_edge("agent", END)
# workflow.add_conditional_edges(
#     "agent",
#     should_continue,
#     {
#         "continue": "tools",
#         "end": END,
#     },
# )
# workflow.add_edge("tools", "agent")

graph = workflow.compile()
