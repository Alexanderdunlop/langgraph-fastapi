from __future__ import annotations

from functools import lru_cache

import langsmith
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from .schemas import GraphConfig
from .settings import SETTINGS

_DEFAULT_DELAY = 60  # seconds


def get_index():
    pc = Pinecone(api_key=SETTINGS.pinecone_api_key)
    return pc.Index(SETTINGS.pinecone_index_name)


@langsmith.traceable
def ensure_configurable(config: RunnableConfig) -> GraphConfig:
    """Merge the user-provided config with default values."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        **GraphConfig(
            delay=configurable.get("delay", _DEFAULT_DELAY),
            model=configurable.get("model", SETTINGS.model),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        ),
    }


@lru_cache
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")
