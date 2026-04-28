from sqlalchemy import String, Text, Integer, ForeignKey, DateTime, Float,Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from pgvector.sqlalchemy import Vector
from app.db.base_class import Base
from datetime import datetime


class Document(Base):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, server_default="text")

    # Extraction pipeline fields
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scanned_pages: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False, server_default="{}")
    extraction_status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")

    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("document.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Extraction pipeline fields
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default="text")
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    section_heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index(
            "chunk_embedding_ivfflat_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 10},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )