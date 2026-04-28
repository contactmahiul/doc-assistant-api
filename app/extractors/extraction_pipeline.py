"""
extraction_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Orchestrates pdf_extractor + table_extractor + ocr_handler into a single
function call that returns everything the chunker needs.

Output shape:
  {
    "metadata": { ... },
    "toc": [ {level, title, page}, ... ],
    "pages": [
      {
        "page_number": 1,
        "blocks": [ {block_type, text, section_heading, ...}, ... ],
        "tables": [ {markdown, dataframe, confidence, ...}, ... ],
        "ocr_text": null | "...",   # only if page was scanned
      },
      ...
    ],
    "full_text": "...",
  }

This dict is also serialisable to JSON (DataFrames are excluded from JSON
output; use the per-page markdown strings for embedding).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.extractors.pdf_extractor import PDFExtractor
from app.extractors.table_extractor import TableExtractor
from app.extractors.ocr_handler import OCRBackend, OCRHandler

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Usage
    -----
    pipeline = ExtractionPipeline()
    result = pipeline.run("paper.pdf")

    # Access structured pages
    for page in result["pages"]:
        print(page["page_number"], len(page["blocks"]), "blocks")
        for tbl in page["tables"]:
            print(tbl["markdown"])

    # Persist for inspection
    pipeline.save_json(result, "output/paper.json")
    """

    def __init__(
        self,
        run_ocr: bool = True,
        ocr_backend: OCRBackend = OCRBackend.TESSERACT,
        ocr_dpi: int = 300,
        ocr_language: str = "eng",
        extract_tables: bool = True,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
        camelot_fallback: bool = True,
    ):
        self.pdf_extractor = PDFExtractor()
        self.table_extractor = TableExtractor(
            min_rows=min_table_rows,
            min_cols=min_table_cols,
            use_camelot_fallback=camelot_fallback,
        ) if extract_tables else None
        self.ocr_handler = OCRHandler(
            backend=ocr_backend, dpi=ocr_dpi, language=ocr_language
        ) if run_ocr else None

    # ── Main entry ───────────────────────────────────────────────────────────

    def run(self, pdf_path: str | Path) -> dict[str, Any]:
        path = Path(pdf_path)
        logger.info("Starting extraction: %s", path.name)

        # 1. Base extraction
        doc = self.pdf_extractor.extract(path)
        logger.info(
            "Pages: %d | Scanned: %s",
            doc.metadata.page_count,
            doc.metadata.scanned_page_numbers,
        )

        # 2. Tables
        tables_by_page: dict[int, list] = {}
        if self.table_extractor:
            all_tables = self.table_extractor.extract(path)
            for t in all_tables:
                tables_by_page.setdefault(t.page_number, []).append(t)
            logger.info("Tables found: %d", len(all_tables))

        # 3. OCR for scanned pages
        ocr_by_page: dict[int, str] = {}
        if self.ocr_handler and doc.metadata.scanned_page_numbers:
            ocr_results = self.ocr_handler.ocr_pages(
                path, doc.metadata.scanned_page_numbers
            )
            for r in ocr_results:
                ocr_by_page[r.page_number] = r.text
            logger.info("OCR'd %d scanned pages", len(ocr_results))

        # 4. Assemble result
        pages_out = []
        for page_idx, blocks in enumerate(doc.pages):
            pn = page_idx + 1
            page_tables = tables_by_page.get(pn, [])

            pages_out.append({
                "page_number": pn,
                "blocks": [self._block_to_dict(b) for b in blocks],
                "tables": [self._table_to_dict(t) for t in page_tables],
                "ocr_text": ocr_by_page.get(pn),
            })

        meta = asdict(doc.metadata)

        return {
            "metadata": meta,
            "toc": doc.toc,
            "pages": pages_out,
            "full_text": doc.full_text,
        }

    # ── Serialisation helpers ────────────────────────────────────────────────

    def save_json(self, result: dict, output_path: str | Path) -> None:
        """Save extraction result to JSON (DataFrames excluded)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Remove DataFrame objects (not JSON serialisable)
        serialisable = json.loads(json.dumps(result, default=str))
        out.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False))
        logger.info("Saved extraction JSON: %s", out)

    @staticmethod
    def _block_to_dict(block) -> dict:
        d = dataclasses.asdict(block)
        d["bbox"] = list(d["bbox"])   # tuple → list for JSON
        return d

    @staticmethod
    def _table_to_dict(table) -> dict:
        return {
            "page_number": table.page_number,
            "table_index": table.table_index,
            "extractor_used": table.extractor_used,
            "markdown": table.markdown,
            "bbox": list(table.bbox) if table.bbox else None,
            "confidence": table.confidence,
            "caption": table.caption,
            # dataframe intentionally excluded from JSON; use markdown for embedding
        }


# ── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python extraction_pipeline.py <pdf_path> [output.json]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "extraction_output.json"

    pipeline = ExtractionPipeline()
    result = pipeline.run(pdf)
    pipeline.save_json(result, out)

    print(f"\n✓ Extracted {result['metadata']['page_count']} pages")
    print(f"  Tables: {sum(len(p['tables']) for p in result['pages'])}")
    print(f"  Scanned pages: {result['metadata']['scanned_page_numbers']}")
    print(f"  Saved → {out}")
