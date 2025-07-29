from pydantic import BaseModel

class QwenMessage(BaseModel):
    role: str
    content: str