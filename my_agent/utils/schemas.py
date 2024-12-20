from __future__ import annotations

from typing import List, Optional

from typing_extensions import TypedDict

class GraphConfig(TypedDict):
    input: str
    chat_history: List[str]
    context: List[str]
    thread_id: str
    user_id: str
    delay: Optional[int]
