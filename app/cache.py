from app.core.redis_client import redis_client
import logging

logger = logging.getLogger(__name__)
def invalidate_product_cache():
    keys = redis_client.keys("products:*")
    if keys:
        redis_client.delete(*keys)
        logger.info(
            "Product cache invalidate"
        )