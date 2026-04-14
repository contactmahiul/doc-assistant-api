import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None

def get_model()->SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: all-MiniLM-L6_v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
    return _model

def embed_text(text:str)->list[float]:
    print(f"DEBUG: Received text value: '{text}' (Type: {type(text)})") # Add thd
    if not text or not text.strip():
        raise ValueError("Can not embed empty text")
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
def embed_batch(texts: list[str])->list[list[float]]:
    if not texts:
        raise ValueError("Cannot embed empty texts")
    model= get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()