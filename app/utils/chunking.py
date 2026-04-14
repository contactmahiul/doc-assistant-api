from transformers import AutoTokenizer 
import logging

logger = logging.getLogger(__name__)

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        logger.info("Loading tokenizer model")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        logger.info ("Tokennizer model loaded successfully")

    return _tokenizer

def chunk_text(text:str, chunk_size: int=200, overlap: int = 50 )-> list[str]:
    if not text or not text.strip():
        raise ValueError("Cannot Chunk empty text")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    step = (chunk_size - overlap)
    chunks = []
    start = 0

    while start < len(token_ids):
        end = start + chunk_size
        chunk_token_ids = token_ids[start:end]
        chunk_text_decoded = tokenizer.decode(chunk_token_ids, skip_special_tokens = False, clean_up_tokenization_spaces=True)
        if chunk_text_decoded.strip():
            chunks.append(chunk_text_decoded.strip())

        start += step

    logger.info(
        f"Chunked text into {len(chunks)}chunks | "
        f"chunk_size={chunk_size},overlap={overlap},"
        f"total_tokens={len(token_ids)}"
    )
    return chunks

def chunk_document(
        documents: list[str],
        chunk_size: int = 200,
        overlap: int = 50
) ->list[list[str]]:
    return [chunk_text(doc, chunk_size, overlap) for doc in documents]


