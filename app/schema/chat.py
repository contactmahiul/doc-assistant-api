# app/schemas/chat.py
from pydantic import BaseModel, Field
from app.schema.query import ChunkResult


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float | None = None  


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChunkResult]
    filtered_count: int