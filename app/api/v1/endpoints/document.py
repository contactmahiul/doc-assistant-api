from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.model.document import Document, Chunk
from app.schema.document import DocumentCreate, DocumentRead, PDFUploadResponse, ExtractionStatus
from app.utils.chunking import chunk_text
from app.utils.pdf import extract_text_from_pdf
from app.utils.embeddings import embed_batch
from app.utils.pdf_chunking import chunk_extracted_result  
import asyncio
from app.core.limiter import limiter
from app.core.redis_client import redis_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=DocumentRead, status_code=201)
@limiter.limit("20/minute")
async def create_document(
    request: Request,
    payload: DocumentCreate,
    db: Session = Depends(get_db)
):
    doc = Document(title=payload.title, content=payload.content)
    db.add(doc)
    db.flush()

    chunks = await asyncio.to_thread(chunk_text, doc.content)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks")

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

    # ── 1. Extract ────────────────────────────────────────────────────────
    try:
        result = await extract_text_from_pdf(file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── 2. Save document with extraction metadata ─────────────────────────
    doc = Document(
        title=title,
        content=result["full_text"],
        source="pdf",
        page_count=result["metadata"]["page_count"],
        scanned_pages=result["metadata"].get("scanned_page_numbers", []),
        extraction_status=ExtractionStatus.PROCESSING,
    )
    db.add(doc)
    db.flush()

    # ── 3. Chunk structured result ────────────────────────────────────────
    chunk_dicts = await asyncio.to_thread(chunk_extracted_result, result)
    if not chunk_dicts:
        raise HTTPException(status_code=422, detail="Could not extract chunks from PDF")

    # ── 4. Embed ──────────────────────────────────────────────────────────
    texts = [c["text"] for c in chunk_dicts]
    vectors = await asyncio.to_thread(embed_batch, texts)

    # ── 5. Store chunks ───────────────────────────────────────────────────
    db.add_all([
        Chunk(
            document_id=doc.id,
            content=c["text"],
            embedding=vector,
            chunk_index=i,
            chunk_type=c["metadata"]["type"],
            page_number=c["metadata"].get("page"),
            section_heading=c["metadata"].get("section"),
            confidence=c["metadata"].get("confidence"),
            bbox=c["metadata"].get("bbox"),
        )
        for i, (c, vector) in enumerate(zip(chunk_dicts, vectors))
    ])

    # ── 6. Mark extraction complete ───────────────────────────────────────
    doc.extraction_status = ExtractionStatus.COMPLETED
    db.commit()
    db.refresh(doc)

    # ── 7. Invalidate cache ───────────────────────────────────────────────
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
        page_count=doc.page_count,
        scanned_pages=doc.scanned_pages,
        extraction_status=doc.extraction_status,
        chunk_count=len(chunk_dicts),
        created_at=doc.created_at,
    )