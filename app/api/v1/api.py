from fastapi import APIRouter
from app.api.v1.endpoints import document,query,chat

api_router = APIRouter()
api_router.include_router( document.router, prefix = "/document",tags=["document"])
api_router.include_router( query.router, prefix="/query", tags=["query"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])