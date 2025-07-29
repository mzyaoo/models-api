from typing import List

from pydantic import BaseModel

from qwen.model import QwenMessage


class QwenRequest(BaseModel):
    stream: bool
    top_p: float
    temperature: float
    max_new_tokens: int
    model: str
    messages: List[QwenMessage]