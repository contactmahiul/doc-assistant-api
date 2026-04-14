from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.schema.query import QueryRequest, QueryResponse, ChunkResult
from app.utils.embeddings import embed_text

router = APIRouter()

@router.post("/", response_model= QueryResponse)
def query_documents(payload: QueryRequest , db: Session = Depends(get_db)):
    query_embed = embed_text(payload.question)
    db.execute(text("SET ivfflat.probes = 3"))

    results = db.execute(text("""
        SELECT
            c.id,
            c.chunk_index,
            c.content,
            c.document_id,
            d.title AS document_title,
            c.embedding<-> CAST(:qvec AS vector) AS distance
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        ORDER BY c.embedding <-> CAST(:qvec AS vector)
        LIMIT :top_k
        """),
        {
        "qvec": str(query_embed),
        "top_k": payload.top_k
        }).fetchall()
    
    response = QueryResponse(
        question= payload.question,
        results = [
            ChunkResult(
                chunk_index= row.chunk_index,
                content= row.content,
                distance= round(row.distance , 4),
                document_id= row.document_id,
                document_title= row.document_title
            )
            for row in results
        ]
    )
    return response

