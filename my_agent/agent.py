from langgraph.graph import StateGraph, END

from my_agent.utils.nodes import call_model, should_continue, tool_node, load_memories
from my_agent.utils.state import AgentState
from my_agent.utils.schemas import GraphConfig

workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("load_memories", load_memories)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("load_memories")
workflow.add_edge("load_memories", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
