from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# --- Chunk Schemas ---

class ChunkBase(BaseModel):
    content: str
    chunk_index: int


class ChunkCreate(ChunkBase):
    document_id: int
    embedding: Optional[list[float]] = None


class ChunkRead(ChunkBase):
    id: int
    document_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Document Schemas ---

class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)


class DocumentCreate(DocumentBase):
    pass


class DocumentRead(DocumentBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    chunks: list[ChunkRead] = []

    model_config = {"from_attributes": True}


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = Field(None, min_length=1)