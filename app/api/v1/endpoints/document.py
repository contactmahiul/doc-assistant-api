from fastapi import APIRouter, Depends, HTTPException,Request, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.model.document import Document, Chunk
from app.schema.document import DocumentCreate, DocumentRead
from app.utils.chunking import chunk_text
from app.utils.embeddings import embed_batch
import asyncio
from app.core.limiter import limiter
from app.core.redis_client import redis_client
import logging
from app.schema.document import PDFUploadResponse
from app.utils.pdf import extract_text_from_pdf

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/",response_model = DocumentRead, status_code=201)
@limiter.limit("20/minute")
async def create_document(request: Request, payload: DocumentCreate, db: Session = Depends(get_db)):
    doc = Document(title = payload.title, content= payload.content)
    db.add(doc)
    db.flush()

    chunks = await asyncio.to_thread(chunk_text, doc.content)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks")
    
    vectors = await asyncio.to_thread(embed_batch, chunks)
    
    db.add_all(
        [
            Chunk(  
                document_id = doc.id,
                content = chunk,
                embedding = vector,
                chunk_index = i
            )
            for i, (chunk, vector) in enumerate(zip(chunks,vectors))
        ]
    )
    db.commit()
    db.refresh(doc)
    try:
        keys = redis_client.keys("response:*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached responses after new document")
    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")
        
    return doc



@router.post("/upload-pdf", response_model=PDFUploadResponse, status_code=201)
@limiter.limit("5/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    db: Session = Depends(get_db)
):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    
    try:
        content, page_count = await extract_text_from_pdf(file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    
    doc = Document(title=title, content=content, source="pdf")
    db.add(doc)
    db.flush()

    
    chunks = await asyncio.to_thread(chunk_text, content)
    if not chunks:
        raise HTTPException(status_code=422, detail="Could not extract chunks from PDF")

    vectors = await asyncio.to_thread(embed_batch, chunks)

    db.add_all([
        Chunk(
            document_id=doc.id,
            content=chunk,
            embedding=vector,
            chunk_index=i
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ])
    db.commit()
    db.refresh(doc)

    
    try:
        keys = redis_client.keys("chat:*") + redis_client.keys("query:*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached responses after PDF upload")
    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")

    return PDFUploadResponse(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        page_count=page_count,
        chunk_count=len(chunks),
        created_at=doc.created_at
    )