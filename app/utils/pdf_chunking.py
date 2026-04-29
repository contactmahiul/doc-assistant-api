from app.utils.chunker import Chunker, ChunkStrategy

_chunker = Chunker(
    strategy=ChunkStrategy.SENTENCE_AWARE,
    chunk_size=400,
    overlap=60,
    respect_headings=True,
    include_tables=True,
)


def chunk_extracted_result(result: dict) -> list[dict]:
    chunks = _chunker.chunk_document(result)

    return [
        {
            "text": c.text,
            "metadata": {
                "type": "table" if c.is_table else (
                    "ocr" if "ocr" in c.block_types else "text"
                ),
                "page": c.page_start,
                "section": c.section_heading,
                "confidence": c.extra.get("confidence"),
                "bbox": None,
            }
        }
        for c in chunks
    ]