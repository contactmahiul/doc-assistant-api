from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schema.chat import ChatRequest, ChatResponse
from app.schema.query import ChunkResult
from app.utils.llm import generate_answer
from app.utils.retrieval import retrieve_relevant_chunks
from app.utils.reranker import rerank                    
import asyncio
from app.core.limiter import limiter
from app.cache import get_cached_response, set_cached_response
from app.utils.faithfulness import faithfulness_check, FALLBACK_ANSWER, FAITHFULNESS_THRESHOLD
import logging


router = APIRouter()
logger = logging.getLogger(__name__)



@router.post("/", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, payload: ChatRequest, db: Session = Depends(get_db)):

    cache_key = f"{payload.question}::{payload.search_mode}"
    cached = get_cached_response(cache_key)
    if cached:
        return ChatResponse(**cached)

    relevant, filtered_count = await retrieve_relevant_chunks(
        question=payload.question,
        db=db,
        top_k=max(payload.top_k * 4, 20),  
        threshold=payload.threshold,
        search_mode=payload.search_mode,
    )

    if not relevant:
        return ChatResponse(
            question=payload.question,
            answer="I could not find relevant information in the documents.",
            search_mode=payload.search_mode,
            filtered_count=filtered_count,
            sources=[],
        )

    top_texts = await rerank(
        question=payload.question,
        chunks=relevant,
        top_n=payload.top_k,                
    )

    reranked_set = set(top_texts)
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

        if row.content not in reranked_set: 
            continue

        sources.append(ChunkResult(
            chunk_index=row.chunk_index,
            content=row.content,
            document_id=row.document_id,
            document_title=row.document_title,
            distance=score_kwargs.get("distance"),
            rrf_score=score_kwargs.get("rrf_score"),
            fts_rank=score_kwargs.get("fts_rank"),
        ))

    answer = await asyncio.to_thread(generate_answer, payload.question, top_texts)
    faith = await faithfulness_check(answer, top_texts)

    if faith["score"] < FAITHFULNESS_THRESHOLD:
        logger.warning(
            f"Faithfulness check failed | score={faith['score']} "
            f"reason='{faith['reason']}' question='{payload.question[:60]}'"
        )
        answer = FALLBACK_ANSWER

    response = ChatResponse(
        question=payload.question,
        answer=answer,
        search_mode=payload.search_mode,
        filtered_count=filtered_count,
        sources=sources,
        faithfulness_score=faith["score"],     
        faithfulness_reason=faith["reason"],
    )

    set_cached_response(cache_key, response.model_dump())
    return response