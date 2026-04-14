from fastapi import FastAPI
from app.db.base import Base
from app.db.session import engine
from app.api.v1 import api
from app.core.logging_config import setup_logging
from app.middleware.correlation import CorrelationIDMiddleware
from fastapi.exceptions import HTTPException, RequestValidationError
from app.core.error_handlers import (
    http_exception_handler,
    validation_exception_handler,
    unexpected_exception_handler
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded 
from app.core.limiter import limiter   

setup_logging()


Base.metadata.create_all(bind=engine)
app = FastAPI()
app.include_router(api.api_router,prefix="/api/v1")
app.add_middleware(CorrelationIDMiddleware)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unexpected_exception_handler)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
@app.get("/")
def home():
    return {"message": "API is running. Visit /docs for the swagger UI."}
    


