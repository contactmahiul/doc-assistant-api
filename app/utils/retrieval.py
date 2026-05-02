import logging
import asyncio
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from app.utils.embeddings import embed_text
from app.cache import get_cached_embedding, set_cached_embedding
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)



def _semantic_query(db: Session, query_vector, top_k: int):
    db.execute(text("SET ivfflat.probes = 3"))
    return db.execute(text("""
        SELECT
            c.id,
            c.chunk_index,
            c.content,
            c.document_id,
            d.title AS document_title,
            c.embedding <=> CAST(:qvec AS vector) AS distance
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :top_k
    """), {"qvec": str(query_vector), "top_k": top_k}).fetchall()


def _fts_query(db: Session, question: str, top_k: int):
    return db.execute(text("""
        SELECT
            c.id,
            c.chunk_index,
            c.content,
            c.document_id,
            d.title AS document_title,
            ts_rank_cd(c.fts_vector, query) AS fts_rank
        FROM chunk c
        JOIN document d ON d.id = c.document_id,
        plainto_tsquery('english', :question) query
        WHERE c.fts_vector @@ query
        ORDER BY fts_rank DESC
        LIMIT :top_k
    """), {"question": question, "top_k": top_k}).fetchall()



def _rrf_fusion(
    semantic_rows: list,
    keyword_rows: list,
    k: int = 60,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    scores: dict = defaultdict(float)
    chunks: dict = {}

    for rank, row in enumerate(semantic_rows):
        scores[row.id] += semantic_weight * (1 / (k + rank + 1))
        chunks[row.id] = row

    for rank, row in enumerate(keyword_rows):
        scores[row.id] += keyword_weight * (1 / (k + rank + 1))
        chunks.setdefault(row.id, row)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"row": chunks[cid], "rrf_score": score} for cid, score in ranked]  



async def retrieve_relevant_chunks(
    question: str,
    db: Session,
    top_k: int = 5,
    threshold: float | None = None,
    search_mode: str = "hybrid",          
) -> tuple[list, int]:

    if not threshold:
        threshold = settings.RETRIEVAL_DISTANCE_THRESHOLD

    query_vector = get_cached_embedding(question)
    if query_vector is None:
        query_vector = await asyncio.to_thread(embed_text, question)
        set_cached_embedding(question, query_vector)

    fetch_k = top_k * 4  
    if search_mode == "semantic":
        rows = await asyncio.to_thread(_semantic_query, db, query_vector, top_k)
        relevant = [row for row in rows if row.distance <= threshold]
        filtered_count = len(rows) - len(relevant)
        logger.info(
            f"Retrieval | mode=semantic question='{question[:60]}' "
            f"returned={len(rows)} relevant={len(relevant)} filtered={filtered_count}"
        )
        return relevant, filtered_count

    elif search_mode == "keyword":
        rows = await asyncio.to_thread(_fts_query, db, question, top_k)
        logger.info(
            f"Retrieval | mode=keyword question='{question[:60]}' returned={len(rows)}"
        )
        return list(rows), 0

    else:  
        def semantic_with_own_session():
            with SessionLocal() as session:
                return _semantic_query(session, query_vector, fetch_k)

        def fts_with_own_session():
            with SessionLocal() as session:
                return _fts_query(session, question, fetch_k)

        semantic_rows, keyword_rows = await asyncio.gather(
            asyncio.to_thread(semantic_with_own_session),
            asyncio.to_thread(fts_with_own_session),
        )

        semantic_rows = [r for r in semantic_rows if r.distance <= threshold]

        fused = _rrf_fusion(semantic_rows, keyword_rows)
        fused = fused[:top_k]

        rrf_threshold = getattr(settings, "RETRIEVAL_RRF_THRESHOLD", 0.002)
        relevant = [r for r in fused if r["rrf_score"] >= rrf_threshold]
        filtered_count = len(fused) - len(relevant)

        logger.info(
            f"Retrieval | mode=hybrid question='{question[:60]}' "
            f"top_k={top_k} rrf_threshold={rrf_threshold} "
            f"semantic={len(semantic_rows)} keyword={len(keyword_rows)} "
            f"fused={len(fused)} relevant={len(relevant)} filtered={filtered_count}"
        )

        return relevant, filtered_count