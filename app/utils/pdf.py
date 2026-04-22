import logging
import fitz
from fastapi import UploadFile

logger = logging.getLogger(__name__)

async def extract_text_from_pdf(file: UploadFile) -> tuple[str, int]:
 
    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise ValueError("Uploaded PDF file is empty")

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    if pdf_document.page_count == 0:
        raise ValueError("PDF has no pages")

    page_count = pdf_document.page_count
    extracted_pages = []

    for page_num in range(page_count):
        page = pdf_document[page_num]
        text = page.get_text()
        if text.strip():
            extracted_pages.append(text.strip())

    pdf_document.close()

    if not extracted_pages:
        raise ValueError("Could not extract any text from PDF — may be scanned/image based")

    full_text = "\n\n".join(extracted_pages)

    logger.info(
        f"PDF extracted | pages={page_count} chars={len(full_text)}"
    )

    return full_text, page_count