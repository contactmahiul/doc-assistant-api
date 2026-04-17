from fastapi import APIRouter, Depends,Request
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.schema.query import QueryRequest, QueryResponse, ChunkResult
from app.utils.embeddings import embed_text
from app.core.config import settings
from app.utils.retrieval import retrieve_relevant_chunks
import asyncio
from app.core.limiter import limiter

router = APIRouter()

@router.post("/", response_model= QueryResponse)
@limiter.limit("10/minute")
async def query_documents(request: Request, payload: QueryRequest , db: Session = Depends(get_db)):

    relevant, filtered_count = await retrieve_relevant_chunks(
        question=payload.question,
        db=db,
        top_k=payload.top_k,
        threshold=payload.threshold
    )

    response = QueryResponse(
        question= payload.question,
        filtered_count=filtered_count,
        results = [
            ChunkResult(
                chunk_index= row.chunk_index,
                content= row.content,
                distance= round(row.distance , 4),
                document_id= row.document_id,
                document_title= row.document_title
            )
            for row in relevant
        ]
    )
    return response

