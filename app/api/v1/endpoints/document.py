from fastapi import APIRouter, Depends, HTTPException,Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.model.document import Document, Chunk
from app.schema.document import DocumentCreate, DocumentRead
from app.utils.chunking import chunk_text
from app.utils.embeddings import embed_batch
import asyncio
from app.core.limiter import limiter

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
    return doc
