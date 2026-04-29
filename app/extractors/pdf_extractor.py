from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  
import pdfplumber



@dataclass
class PageBlock:
    block_type: str          
    text: str
    page_number: int         
    bbox: tuple[float, float, float, float]   
    font_size: float
    font_name: str
    is_bold: bool
    section_heading: Optional[str] = None  


@dataclass
class DocumentMetadata:
   
    file_path: str
    file_name: str
    file_hash_sha256: str
    file_size_bytes: int
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    has_scanned_pages: bool = False
    scanned_page_numbers: list[int] = field(default_factory=list)
    language_hint: Optional[str] = None


@dataclass
class ExtractedDocument:
    
    metadata: DocumentMetadata
    pages: list[list[PageBlock]]   
    full_text: str                 
    toc: list[dict]               



_SCANNED_CHARS_PER_PAGE_THRESHOLD = 100  
_HEADING_FONT_RATIO = 1.15                


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_pdf_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip().lstrip("D:").replace("'", "")
    for fmt in ("%Y%m%d%H%M%S%z", "%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(raw[:len(fmt.replace("%", "XX"))], fmt).isoformat()
        except ValueError:
            continue
    return raw  


def _classify_block(
    span_sizes: list[float],
    body_font_size: float,
    text: str,
) -> str:
    if not text.strip():
        return "paragraph"
    avg_size = sum(span_sizes) / len(span_sizes) if span_sizes else body_font_size
    if avg_size >= body_font_size * _HEADING_FONT_RATIO:
        return "heading"
    stripped = text.strip()
    if re.match(r"^(\d+\.|[•·▪▸◦\-])\s", stripped):
        return "list_item"
    if len(stripped) < 80 and stripped.endswith((".", "…")) is False and avg_size < body_font_size:
        return "caption"
    return "paragraph"


def _estimate_body_font_size(page: fitz.Page) -> float:
    sizes: dict[float, int] = {}
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                s = round(span.get("size", 0), 1)
                sizes[s] = sizes.get(s, 0) + len(span.get("text", ""))
    if not sizes:
        return 11.0
    return max(sizes, key=sizes.__getitem__)



class PDFExtractor:


    def __init__(
        self,
        scanned_threshold: int = _SCANNED_CHARS_PER_PAGE_THRESHOLD,
        heading_ratio: float = _HEADING_FONT_RATIO,
    ):
        self.scanned_threshold = scanned_threshold
        self.heading_ratio = heading_ratio

    

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        fitz_doc = fitz.open(str(path))
        metadata = self._build_metadata(path, fitz_doc)
        toc = self._extract_toc(fitz_doc)
        pages, scanned_pages = self._extract_pages(fitz_doc)

        metadata.has_scanned_pages = bool(scanned_pages)
        metadata.scanned_page_numbers = scanned_pages

        full_text = self._assemble_full_text(pages)
        fitz_doc.close()

        return ExtractedDocument(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            toc=toc,
        )


    def _build_metadata(self, path: Path, doc: fitz.Document) -> DocumentMetadata:
        info = doc.metadata or {}
        return DocumentMetadata(
            file_path=str(path),
            file_name=path.name,
            file_hash_sha256=_sha256(path),
            file_size_bytes=path.stat().st_size,
            page_count=doc.page_count,
            title=info.get("title") or None,
            author=info.get("author") or None,
            subject=info.get("subject") or None,
            creator=info.get("creator") or None,
            producer=info.get("producer") or None,
            creation_date=_parse_pdf_date(info.get("creationDate")),
            modification_date=_parse_pdf_date(info.get("modDate")),
        )


    def _extract_toc(self, doc: fitz.Document) -> list[dict]:
        raw = doc.get_toc(simple=False)  
        return [
            {"level": entry[0], "title": entry[1].strip(), "page": entry[2]}
            for entry in raw
            if entry[1].strip()
        ]


    def _extract_pages(
        self, doc: fitz.Document
    ) -> tuple[list[list[PageBlock]], list[int]]:
        all_pages: list[list[PageBlock]] = []
        scanned: list[int] = []

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_number = page_idx + 1

            raw_text = page.get_text("text")
            if len(raw_text.strip()) < self.scanned_threshold:
                scanned.append(page_number)
                all_pages.append([])   
                continue

            blocks = self._extract_page_blocks(page, page_number)
            all_pages.append(blocks)

        return all_pages, scanned

    def _extract_page_blocks(
        self, page: fitz.Page, page_number: int
    ) -> list[PageBlock]:
        body_size = _estimate_body_font_size(page)
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        result: list[PageBlock] = []
        current_heading: str | None = None

        for block in raw.get("blocks", []):
            if block.get("type") != 0:  
                continue

            lines = block.get("lines", [])
            if not lines:
                continue

            
            block_text_parts: list[str] = []
            span_sizes: list[float] = []
            dominant_font = ""
            is_bold = False
            bbox = block.get("bbox", (0, 0, 0, 0))

            for line in lines:
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    if not t.strip():
                        continue
                    block_text_parts.append(t)
                    span_sizes.append(span.get("size", body_size))
                    flags = span.get("flags", 0)
                    is_bold = is_bold or bool(flags & 2**4)   
                    if not dominant_font:
                        dominant_font = span.get("font", "")

            text = " ".join(block_text_parts).strip()
            text = re.sub(r"\s+", " ", text)
            if not text:
                continue

            block_type = _classify_block(span_sizes, body_size, text)
            avg_font = sum(span_sizes) / len(span_sizes) if span_sizes else body_size

            if block_type == "heading":
                current_heading = text

            pb = PageBlock(
                block_type=block_type,
                text=text,
                page_number=page_number,
                bbox=tuple(bbox),
                font_size=round(avg_font, 2),
                font_name=dominant_font,
                is_bold=is_bold,
                section_heading=current_heading if block_type != "heading" else None,
            )
            result.append(pb)

        return result

  

    def _assemble_full_text(self, pages: list[list[PageBlock]]) -> str:
        parts: list[str] = []
        for page_blocks in pages:
            for block in page_blocks:
                parts.append(block.text)
        return "\n\n".join(parts)
