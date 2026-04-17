from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.schema.query import QueryRequest, QueryResponse, ChunkResult
from app.utils.embeddings import embed_text
from app.core.config import settings
from app.utils.retrieval import retrieve_relevant_chunks

router = APIRouter()

@router.post("/", response_model= QueryResponse)
def query_documents(payload: QueryRequest , db: Session = Depends(get_db)):
    threshold = payload.threshold if payload.threshold is not None else settings.RETRIEVAL_DISTANCE_THRESHOLD
    query_embed = embed_text(payload.question)
    db.execute(text("SET ivfflat.probes = 3"))



    relevant, filtered_count = retrieve_relevant_chunks(
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

