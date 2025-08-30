from pydantic import BaseModel, Field
from uuid import UUID

class ChatCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)

class ChatOut(BaseModel):
    chat_id: UUID