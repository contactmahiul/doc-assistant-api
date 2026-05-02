import logging
import asyncio
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)
_ranker = None

def get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(
            model_name="ms-marco-MultiBERT-L-12",
            cache_dir="/tmp/flashrank"
        )
    return _ranker

def _rerank_sync(question: str, chunks: list, top_n: int) -> list[str]:
    if not chunks:
        return []

    def get_content(c):
        if isinstance(c, dict):      
            return c["row"].content
        return c.content            

    passages = [
        {"id": i, "text": get_content(c)}
        for i, c in enumerate(chunks)
    ]

    ranker = get_ranker()
    request = RerankRequest(query=question, passages=passages)
    results = ranker.rerank(request)

    reranked = [r["text"] for r in results[:top_n]]
    logger.info(
        f"Rerank | question='{question[:60]}' "
        f"input={len(chunks)} output={len(reranked)}"
    )
    return reranked


async def rerank(question: str, chunks: list, top_n: int = 5) -> list[str]:
    return await asyncio.to_thread(_rerank_sync, question, chunks, top_n)