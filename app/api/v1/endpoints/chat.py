# app/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.schema.chat import ChatRequest, ChatResponse
from app.schema.query import ChunkResult
from app.utils.embeddings import embed_text
from app.utils.llm import generate_answer

router = APIRouter()


@router.post("/", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    # 1. Embed the question
    query_vector = embed_text(payload.question)

    # 2. Retrieve relevant chunks
    db.execute(text("SET ivfflat.probes = 3"))
    rows = db.execute(text("""
        SELECT
            c.id,
            c.chunk_index,
            c.content,
            c.document_id,
            d.title AS document_title,
            c.embedding <-> CAST(:qvec AS vector) AS distance
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        ORDER BY c.embedding <-> CAST(:qvec AS vector)
        LIMIT :top_k
    """), {
        "qvec": str(query_vector),
        "top_k": payload.top_k
    }).fetchall()

    # 3. Extract chunk texts for prompt
    chunk_texts = [row.content for row in rows]

    # 4. Generate answer
    answer = generate_answer(payload.question, chunk_texts)

    # 5. Build response with sources
    return ChatResponse(
        question=payload.question,
        answer=answer,
        sources=[
            ChunkResult(
                chunk_index=row.chunk_index,
                content=row.content,
                document_id=row.document_id,
                document_title=row.document_title,
                distance=round(row.distance, 4)
            )
            for row in rows
        ]
    )