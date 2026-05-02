from pydantic import BaseModel, Field
from typing import Literal
from app.schema.query import ChunkResult

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float | None = None
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"  

class ChatResponse(BaseModel):
    question: str
    answer: str
    search_mode: str          
    sources: list[ChunkResult]
    filtered_count: int