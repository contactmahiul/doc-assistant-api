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

@router.post("/", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_documents(
    request: Request,
    payload: QueryRequest,
    db: Session = Depends(get_db)
):
    relevant, filtered_count = await retrieve_relevant_chunks(
        question=payload.question,
        db=db,
        top_k=payload.top_k,
        threshold=payload.threshold,
        search_mode=payload.search_mode,   
    )

    results = []

    for item in relevant:
        if payload.search_mode == "hybrid":
            row = item["row"]
            results.append(ChunkResult(
                chunk_index=row.chunk_index,
                content=row.content,
                rrf_score=round(item["rrf_score"], 6),
                distance=None,
                fts_rank=None,
                document_id=row.document_id,
                document_title=row.document_title,
            ))

        elif payload.search_mode == "semantic":
            results.append(ChunkResult(
                chunk_index=item.chunk_index,
                content=item.content,
                distance=round(item.distance, 4),
                rrf_score=None,
                fts_rank=None,
                document_id=item.document_id,
                document_title=item.document_title,
            ))

        elif payload.search_mode == "keyword":
            results.append(ChunkResult(
                chunk_index=item.chunk_index,
                content=item.content,
                fts_rank=round(item.fts_rank, 6),
                distance=None,
                rrf_score=None,
                document_id=item.document_id,
                document_title=item.document_title,
            ))

    return QueryResponse(
        question=payload.question,
        search_mode=payload.search_mode,   
        filtered_count=filtered_count,
        results=results,
    )
