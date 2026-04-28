from app.utils.chunking import chunk_text


def chunk_extracted_result(result: dict) -> list[dict]:
    """
    Convert ExtractionPipeline result into flat list of
    { text, metadata } dicts ready for embedding.
    """
    chunks = []

    for page in result["pages"]:
        pn = page["page_number"]

        # ── Text blocks ───────────────────────────────────────────────
        for block in page["blocks"]:
            text = block.get("text", "").strip()
            if not text:
                continue
            # Run existing chunker on each block to respect chunk size
            sub_chunks = chunk_text(text, chunk_size=200, overlap=50)
            for sub in sub_chunks:
                chunks.append({
                    "text": sub,
                    "metadata": {
                        "type": "text",
                        "page": pn,
                        "section": block.get("section_heading"),
                        "confidence": None,
                        "bbox": None,
                    }
                })

        # ── Tables ────────────────────────────────────────────────────
        for table in page["tables"]:
            markdown = table.get("markdown", "").strip()
            if not markdown:
                continue
            chunks.append({
                "text": markdown,
                "metadata": {
                    "type": "table",
                    "page": pn,
                    "section": None,
                    "confidence": table.get("confidence"),
                    "bbox": table.get("bbox"),
                }
            })

        # ── OCR text ──────────────────────────────────────────────────
        ocr_text = page.get("ocr_text")
        if ocr_text and ocr_text.strip():
            sub_chunks = chunk_text(ocr_text, chunk_size=200, overlap=50)
            for sub in sub_chunks:
                chunks.append({
                    "text": sub,
                    "metadata": {
                        "type": "ocr",
                        "page": pn,
                        "section": None,
                        "confidence": None,
                        "bbox": None,
                    }
                })

    return chunks