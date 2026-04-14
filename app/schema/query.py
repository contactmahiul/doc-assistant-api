from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ChunkResult(BaseModel):
    chunk_index: int
    content: str
    distance: float
    document_id: int
    document_title: str

    model_config = {"from_attributes": True}


class QueryResponse(BaseModel):
    question: str
    results: list[ChunkResult]