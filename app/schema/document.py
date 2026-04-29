from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum



class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    OCR = "ocr"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"



class ChunkBase(BaseModel):
    content: str
    chunk_index: int


class ChunkCreate(ChunkBase):
    document_id: int
    embedding: Optional[list[float]] = None
    chunk_type: ChunkType = ChunkType.TEXT
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[dict] = None


class ChunkRead(ChunkBase):
    id: int
    document_id: int
    created_at: datetime
    chunk_type: ChunkType
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[dict] = None

    model_config = {"from_attributes": True}



class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)


class DocumentCreate(DocumentBase):
    pass


class DocumentRead(DocumentBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    source: str
    page_count: Optional[int] = None
    scanned_pages: list[int] = []
    extraction_status: ExtractionStatus
    chunks: list[ChunkRead] = []

    model_config = {"from_attributes": True}


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = Field(None, min_length=1)



class PDFUploadResponse(BaseModel):
    id: int
    title: str
    source: str
    page_count: Optional[int] = None
    scanned_pages: list[int] = []
    extraction_status: ExtractionStatus
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}



class DocumentExtractionUpdate(BaseModel):
    page_count: Optional[int] = None
    scanned_pages: list[int] = []
    extraction_status: ExtractionStatus