
from fastapi import APIRouter, Depends,Request
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.schema.chat import ChatRequest, ChatResponse
from app.schema.query import ChunkResult
from app.utils.embeddings import embed_text
from app.utils.llm import generate_answer
from app.utils.retrieval import retrieve_relevant_chunks
import asyncio
from app.core.limiter import limiter
from app.cache import get_cached_response, set_cached_response

router = APIRouter()


@router.post("/", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, payload: ChatRequest, db: Session = Depends(get_db)):

    cached = get_cached_response(payload.question)
    if cached:
        return ChatResponse(**cached)

    relevant, filtered_count = await retrieve_relevant_chunks(
        question=payload.question,
        db=db,
        top_k=payload.top_k,
        threshold=payload.threshold,
        search_mode=payload.search_mode,
    )   

    chunk_texts = []
    sources = []

    for item in relevant:
        if payload.search_mode == "hybrid":
            row = item["row"]                          
            score_kwargs = {"rrf_score": round(item["rrf_score"], 6)}
        elif payload.search_mode == "keyword":
            row = item                                 
            score_kwargs = {"fts_rank": round(item.fts_rank, 6)}
        else: 
            row = item                                 
            score_kwargs = {"distance": round(item.distance, 4)}

        chunk_texts.append(row.content)               

        sources.append(ChunkResult(
            chunk_index=row.chunk_index,
            content=row.content,
            document_id=row.document_id,
            document_title=row.document_title,
            distance=score_kwargs.get("distance"),
            rrf_score=score_kwargs.get("rrf_score"),
            fts_rank=score_kwargs.get("fts_rank"),
        ))

    answer = await asyncio.to_thread(generate_answer, payload.question, chunk_texts)

    response = ChatResponse(
        question=payload.question,
        answer=answer,
        search_mode=payload.search_mode,               
        filtered_count=filtered_count,
        sources=sources,
    )

    set_cached_response(payload.question, response.model_dump())

    return response
    