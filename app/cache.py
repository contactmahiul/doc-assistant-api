from app.core.redis_client import redis_client
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

def make_key(prefix: str, value: str )->str:
    hashed = hashlib.md5(value.encode()).hexdigest()
    return f"{prefix}:{hashed}"

def set_cached_embedding(text: str , embedding: list[float], ttl: int = 86400):
    key = make_key("embedding", text)
    try:
        redis_client.setex(key, ttl, json.dumps(embedding))
        logger.info(f"Embedding cache SET | key = {key} ttl = {ttl}")
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")

def get_cached_embedding(text: str)-> list[float] | None:
    key = make_key("embedding", text)
    try:
        cached = redis_client.get(key)
        if cached:
            logger.info(f"Embedding cache HIT | key = {key}")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
        return None
    
def get_cached_response(question: str) -> dict | None:
    key = make_key("response", question)
    try:
        cached = redis_client.get(key)
        if cached:
            logger.info(f"Response cache HIT | key={key}")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
    return None

def set_cached_response(question: str, response: dict, ttl: int = 3600):
    key = make_key("response", question)
    try:
        redis_client.setex(key, ttl, json.dumps(response))
        logger.info(f"Response cache SET | key={key} ttl={ttl}")
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")


