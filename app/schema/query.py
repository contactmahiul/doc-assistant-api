from pydantic import BaseModel, Field
from typing import Literal


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    threshold: float | None = None
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid" 

class ChunkResult(BaseModel):
    chunk_index: int
    content: str
    distance: float | None = None     
    rrf_score: float | None = None    
    fts_rank: float | None = None    
    document_id: int
    document_title: str


class QueryResponse(BaseModel):
    question: str
    search_mode: str                  
    filtered_count: int
    results: list[ChunkResult]