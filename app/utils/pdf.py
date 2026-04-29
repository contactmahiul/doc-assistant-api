import logging
import tempfile
import os
from fastapi import UploadFile
from app.extractors.extraction_pipeline import ExtractionPipeline
from app.extractors.ocr_handler import OCRBackend

logger = logging.getLogger(__name__)

_pipeline = ExtractionPipeline(
    run_ocr=True,
    ocr_backend=OCRBackend.TESSERACT,
    ocr_dpi=300,
    ocr_language="eng",
    extract_tables=True,
    min_table_rows=2,
    min_table_cols=2,
    camelot_fallback=True,
)


async def extract_text_from_pdf(file: UploadFile) -> dict:

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise ValueError("Uploaded PDF file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = _pipeline.run(tmp_path)
    except Exception as e:
        logger.error(f"Extraction pipeline failed: {e}")
        raise ValueError(f"Failed to extract content from PDF: {e}")
    finally:
        os.unlink(tmp_path)

    total_content = _count_extractable_content(result)
    if total_content == 0:
        raise ValueError("Could not extract any content from PDF")

    logger.info(
        f"PDF extracted | file={file.filename} "
        f"pages={result['metadata']['page_count']} "
        f"scanned={result['metadata']['scanned_page_numbers']} "
        f"tables={sum(len(p['tables']) for p in result['pages'])} "
        f"chars={len(result['full_text'])}"
    )

    return result


def _count_extractable_content(result: dict) -> int:
    total = 0
    for page in result["pages"]:
        total += len([b for b in page["blocks"] if b["text"].strip()])
        total += len(page["tables"])
        if page["ocr_text"]:
            total += 1
    return total