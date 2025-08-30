# app/schemas/source.py
from pydantic import BaseModel, Field
from typing import Literal

# existentes
class SourceCreate(BaseModel):
    type: Literal["file","text"]
    name: str

class SourceCreateText(SourceCreate):
    type: Literal["text"]
    content: str

class SourceCreateFile(SourceCreate):
    type: Literal["file"]
    file_url: str
    file_name: str
    file_size: int

class SourceOut(BaseModel):
    source_id: str
    status: str

# nuevos
class SourceInitIn(BaseModel):
    name: str | None = Field(default=None, description="Nombre lógico de la fuente (opcional)")

class SourceConfirmUploadIn(BaseModel):
    file_url: str
    file_name: str
    file_size: int