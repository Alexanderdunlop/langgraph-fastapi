from functools import lru_cache
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from datetime import datetime, timezone
from my_agent.utils.state import AgentState
from langchain_core.runnables.config import (
    RunnableConfig,
    get_executor_for_config,
)
from .pinecone import ensure_configurable
from .tools import search_memory, fetch_core_memories
from langchain_core.messages.utils import get_buffer_string
import tiktoken

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

def should_continue(state):
    """Determine whether to use tools or end the conversation based on the last message.
    
    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        str: "end" if the conversation should end, "continue" if tools should be used.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


system_prompt = (
    "You are a helpful assistant with advanced long-term memory"
    " capabilities. Powered by a stateless LLM, you must rely on"
    " external memory to store information between conversations."
    " Utilize the available memory tools to store and retrieve"
    " important details that will help you better attend to the user's"
    " needs and understand their context.\n\n"
    "Memory Usage Guidelines:\n"
    "1. Actively use memory tools (save_core_memory, save_recall_memory)"
    " to build a comprehensive understanding of the user.\n"
    "2. Make informed suppositions and extrapolations based on stored"
    " memories.\n"
    "3. Regularly reflect on past interactions to identify patterns and"
    " preferences.\n"
    "4. Update your mental model of the user with each new piece of"
    " information.\n"
    "5. Cross-reference new information with existing memories for"
    " consistency.\n"
    "6. Prioritize storing emotional context and personal values"
    " alongside facts.\n"
    "7. Use memory to anticipate needs and tailor responses to the"
    " user's style.\n"
    "8. Recognize and acknowledge changes in the user's situation or"
    " perspectives over time.\n"
    "9. Leverage memories to provide personalized examples and"
    " analogies.\n"
    "10. Recall past challenges or successes to inform current"
    " problem-solving.\n\n"
    "## Core Memories\n"
    "Core memories are fundamental to understanding the user and are"
    " always available:\n{core_memories}\n\n"
    "## Recall Memories\n"
    "Recall memories are contextually retrieved based on the current"
    " conversation:\n{recall_memories}\n\n"
    "## Instructions\n"
    "Engage with the user naturally, as a trusted colleague or friend."
    " There's no need to explicitly mention your memory capabilities."
    " Instead, seamlessly incorporate your understanding of the user"
    " into your responses. Be attentive to subtle cues and underlying"
    " emotions. Adapt your communication style to match the user's"
    " preferences and current emotional state. Use tools to persist"
    " information you want to retain in the next conversation. If you"
    " do call tools, all text preceding the tool call is an internal"
    " message. Respond AFTER calling the tool, once you have"
    " confirmation that the tool completed successfully.\n\n"
    "Current system time: {current_time}\n\n",
)

def call_model(state: AgentState, config: RunnableConfig):
    """Process the current state and generate a response using the LLM."""

    messages = state["messages"]

    # Load memories
    core_str = (
        "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    
    # Get current time
    current_time = datetime.now(tz=timezone.utc).isoformat()
    
    # Format system prompt with memories and time
    formatted_system_prompt = str(system_prompt).format(
        core_memories=core_str,
        recall_memories=recall_str,
        current_time=current_time
    )

    messages = [{"role": "system", "content": formatted_system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"messages": [response]}

def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
    """Load core and recall memories for the current conversation.

    Args:
        state (AgentState): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """
    configurable = ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


tool_node = ToolNode(tools)