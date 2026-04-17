
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

router = APIRouter()


@router.post("/", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request,payload: ChatRequest, db: Session = Depends(get_db)):
   
    relevant, filtered_count = await retrieve_relevant_chunks(
        question=payload.question,
        db=db,
        top_k=payload.top_k,
        threshold=payload.threshold
    )

    chunk_texts = [row.content for row in relevant]
    answer = await asyncio.to_thread(generate_answer, payload.question, chunk_texts)

    return ChatResponse(
        question=payload.question,
        answer=answer,
        filtered_count = filtered_count,
        sources=[
            ChunkResult(
                chunk_index=row.chunk_index,
                content=row.content,
                document_id=row.document_id,
                document_title=row.document_title,
                distance=round(row.distance, 4)
            )
            for row in relevant
        ]
    )