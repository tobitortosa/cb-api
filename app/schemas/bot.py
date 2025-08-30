from pydantic import BaseModel, Field

class BotCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    system_prompt: str | None = None

class BotOut(BaseModel):
    bot_id: str