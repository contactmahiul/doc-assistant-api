from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

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
    source: str 

    model_config = {"from_attributes": True}


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = Field(None, min_length=1)

class PDFUploadResponse(BaseModel):
    id: int
    title: str
    source: str
    page_count: int
    chunk_count: int
    created_at: datetime
    model_config = {"from_attributes": True}