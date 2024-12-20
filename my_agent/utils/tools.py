# from langchain_community.tools.tavily_search import TavilySearchResults

# tools = [TavilySearchResults(max_results=1)]
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import langsmith
from langchain_core.tools import tool
from langchain_core.runnables.config import ensure_config

from .pinecone import ensure_configurable, get_embeddings, get_index
from .settings import SETTINGS
from .constants import TYPE_KEY, PAYLOAD_KEY, PATCH_PATH, PATH_KEY, TIMESTAMP_KEY, INSERT_PATH


_EMPTY_VEC = [0.0] * 1536

@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    config = ensure_config()
    configurable = ensure_configurable(config)
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(memory)
    current_time = datetime.now(tz=timezone.utc).isoformat()
    path = INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=str(uuid.uuid4()),
    )
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                PAYLOAD_KEY: memory,
                PATH_KEY: path,
                TIMESTAMP_KEY: current_time,
                TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    get_index().upsert(
        vectors=documents,
        namespace=SETTINGS.pinecone_namespace,
    )
    return memory


@tool
def search_memory(query: str, top_k: int = 5) -> list[str]:
    """Search for memories in the database based on semantic similarity."""
    config = ensure_config()
    configurable = ensure_configurable(config)
    embeddings = get_embeddings()
    vector = embeddings.embed_query(query)
    with langsmith.trace("query", inputs={"query": query, "top_k": top_k}) as rt:
        response = get_index().query(
            vector=vector,
            filter={
                "user_id": {"$eq": configurable["user_id"]},
                TYPE_KEY: {"$eq": "recall"},
            },
            namespace=SETTINGS.pinecone_namespace,
            include_metadata=True,
            top_k=top_k,
        )
        rt.end(outputs={"response": response})
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"][PAYLOAD_KEY] for m in matches]
    return memories


@langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, list[str]]: The path and list of core memories.
    """
    path = PATCH_PATH.format(user_id=user_id)
    response = get_index().fetch(
        ids=[path], namespace=SETTINGS.pinecone_namespace
    )
    memories = []
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories

@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory in the database.

    Args:
        memory (str): The memory to store.
        index (Optional[int]): The index at which to store the memory.

    Returns:
        str: A confirmation message.
    """
    config = ensure_config()
    configurable = ensure_configurable(config)
    path, memories = fetch_core_memories(configurable["user_id"])
    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)
    documents = [
        {
            "id": path,
            "values": _EMPTY_VEC,
            "metadata": {
                PAYLOAD_KEY: json.dumps({"memories": memories}),
                PATH_KEY: path,
                TIMESTAMP_KEY: datetime.now(tz=timezone.utc).isoformat(),
                TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    get_index().upsert(
        vectors=documents,
        namespace=SETTINGS.pinecone_namespace,
    )
    return "Memory stored."


tools = [save_recall_memory, search_memory, store_core_memory]