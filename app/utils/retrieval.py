import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
import asyncio

from app.core.config import settings
from app.utils.embeddings import embed_text
from app.cache import get_cached_embedding,set_cached_embedding

logger = logging.getLogger(__name__)

async def retrieve_relevant_chunks(
    question: str,
    db: Session,
    top_k: int = 5,
    threshold: float | None = None
) -> tuple[list, int]:

    threshold = threshold if threshold is not None else settings.RETRIEVAL_DISTANCE_THRESHOLD

    query_vector = get_cached_embedding(question)
    if query_vector is None:
        query_vector = await asyncio.to_thread(embed_text, question)

        set_cached_embedding(question, query_vector)
    def _db_query():
        db.execute(text("SET ivfflat.probes = 3"))
        return db.execute(text("""
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
            "top_k": top_k
        }).fetchall()

    rows = await asyncio.to_thread(_db_query)
    relevant = [row for row in rows if row.distance <= threshold]
    filtered_count = len(rows) - len(relevant)

    logger.info(
        f"Retrieval | question='{question[:60]}' "
        f"top_k={top_k} threshold={threshold} "
        f"returned={len(rows)} relevant={len(relevant)} filtered={filtered_count}"
    )

    return relevant, filtered_count