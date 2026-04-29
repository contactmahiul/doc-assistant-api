from __future__ import annotations
import hashlib
import re
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

def _load_sentence_splitter():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
        nlp.enable_pipe("senter")
        def _split(text: str) -> list[str]:
            return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        return _split
    except Exception:
        pass
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return lambda text: [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        pass
   
    _SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    return lambda text: [s.strip() for s in _SENT_RE.split(text) if s.strip()]


_split_sentences = _load_sentence_splitter()




def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)



class ChunkStrategy(str, Enum):
    SENTENCE_AWARE = "sentence_aware"
    PARAGRAPH      = "paragraph"
    FIXED_TOKEN    = "fixed_token"


@dataclass
class Chunk:


    chunk_id: str                   
    doc_id: str                     
    chunk_index: int                
    total_chunks: int              
    strategy: str


    text: str                     
    token_count: int


    source_file: str
    page_start: int                 
    page_end: int                   
    section_heading: Optional[str]  
    block_types: list[str]          
    is_table: bool


    overlap_tokens_prev: int = 0   
    overlap_tokens_next: int = 0  

  
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def make_id(text: str, doc_id: str, index: int) -> str:
        h = hashlib.sha256(f"{doc_id}::{index}::{text[:200]}".encode()).hexdigest()
        return str(uuid.UUID(h[:32]))



class Chunker:
    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE_AWARE,
        chunk_size: int = 400,
        overlap: int = 60,
        min_chunk_size: int = 50,
        respect_headings: bool = True,   
        respect_pages: bool = False,     
        include_tables: bool = True,    
        table_max_tokens: int = 800,    
    ):
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_headings = respect_headings
        self.respect_pages = respect_pages
        self.include_tables = include_tables
        self.table_max_tokens = table_max_tokens


    def chunk_document(self, extraction_result: dict) -> list[Chunk]:
        meta = extraction_result["metadata"]
        doc_id = meta["file_hash_sha256"]
        source_file = meta["file_name"]
        pages = extraction_result["pages"]

        raw_chunks: list[Chunk] = []

        for page in pages:
            pn = page["page_number"]
            ocr_text = page.get("ocr_text")

            if ocr_text and not page["blocks"]:
                raw_chunks.extend(
                    self._chunk_plain_text(
                        ocr_text, doc_id, source_file, pn, pn,
                        section_heading=None, block_types=["ocr"],
                    )
                )
                continue

            
            if self.include_tables:
                for tbl in page.get("tables", []):
                    raw_chunks.extend(
                        self._chunk_table(tbl, doc_id, source_file, pn)
                    )

            
            if self.strategy == ChunkStrategy.PARAGRAPH:
                raw_chunks.extend(
                    self._chunk_by_paragraph(page["blocks"], doc_id, source_file, pn)
                )
            elif self.strategy == ChunkStrategy.FIXED_TOKEN:
                raw_chunks.extend(
                    self._chunk_fixed_token(page["blocks"], doc_id, source_file, pn)
                )
            else:
                raw_chunks.extend(
                    self._chunk_sentence_aware(page["blocks"], doc_id, source_file, pn)
                )

        return self._finalise(raw_chunks, doc_id)


    def _chunk_sentence_aware(
        self,
        blocks: list[dict],
        doc_id: str,
        source_file: str,
        page_number: int,
    ) -> list[Chunk]:

        sentence_stream: list[tuple[str, Optional[str], str]] = []
        current_heading: Optional[str] = None

        for block in blocks:
            btype = block["block_type"]
            text  = block["text"].strip()
            if not text:
                continue

            if btype == "heading":
                current_heading = text
                sentence_stream.append((text, None, "heading"))
                continue

            heading = block.get("section_heading") or current_heading
            sentences = _split_sentences(text)
            for s in sentences:
                if s:
                    sentence_stream.append((s, heading, btype))

        if not sentence_stream:
            return []

        chunks: list[Chunk] = []
        i = 0

        while i < len(sentence_stream):
            window: list[tuple[str, Optional[str], str]] = []
            token_count = 0
            last_heading: Optional[str] = None
            btypes_seen: set[str] = set()

            # if (
            #     self.respect_headings
            #     and sentence_stream[i][2] == "heading"
            #     and window
            # ):
            #     pass 
            
            j = i
            while j < len(sentence_stream):
                sent, hdg, btype = sentence_stream[j]
                sent_tokens = _approx_tokens(sent)

                if (
                    self.respect_headings
                    and btype == "heading"
                    and window
                ):
                    break

                if token_count + sent_tokens > self.chunk_size and window:
                    break

                window.append((sent, hdg, btype))
                token_count += sent_tokens
                if hdg:
                    last_heading = hdg
                btypes_seen.add(btype)
                j += 1

            if not window:
                window = [sentence_stream[i]]
                j = i + 1

            chunk_text = " ".join(s for s, _, _ in window)

            overlap_prefix = ""
            overlap_tok = 0
            if chunks and not chunks[-1].is_table:
                prev_sents = chunks[-1].text.split(". ")
                acc = ""
                for ps in reversed(prev_sents):
                    cand = ps + (". " if not ps.endswith(".") else " ")
                    if _approx_tokens(acc + cand) <= self.overlap:
                        acc = cand + acc
                    else:
                        break
                if acc.strip():
                    overlap_prefix = acc.strip() + " "
                    overlap_tok = _approx_tokens(overlap_prefix)

            full_text = (overlap_prefix + chunk_text).strip()

            c = Chunk(
                chunk_id="",
                doc_id=doc_id,
                chunk_index=0,
                total_chunks=0,
                strategy=ChunkStrategy.SENTENCE_AWARE,
                text=full_text,
                token_count=_approx_tokens(full_text),
                source_file=source_file,
                page_start=page_number,
                page_end=page_number,
                section_heading=last_heading,
                block_types=sorted(btypes_seen),
                is_table=False,
                overlap_tokens_prev=overlap_tok,
            )
            chunks.append(c)
            i = j

        return self._merge_tiny(chunks)


    def _chunk_by_paragraph(
        self,
        blocks: list[dict],
        doc_id: str,
        source_file: str,
        page_number: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        pending_text = ""
        pending_heading: Optional[str] = None
        pending_btypes: set[str] = set()

        def flush():
            nonlocal pending_text, pending_btypes
            if pending_text.strip():
                chunks.append(Chunk(
                    chunk_id="",
                    doc_id=doc_id,
                    chunk_index=0,
                    total_chunks=0,
                    strategy=ChunkStrategy.PARAGRAPH,
                    text=pending_text.strip(),
                    token_count=_approx_tokens(pending_text),
                    source_file=source_file,
                    page_start=page_number,
                    page_end=page_number,
                    section_heading=pending_heading,
                    block_types=sorted(pending_btypes),
                    is_table=False,
                ))
            pending_text = ""
            pending_btypes = set()

        for block in blocks:
            btype = block["block_type"]
            text  = block["text"].strip()
            if not text:
                continue

            toks = _approx_tokens(text)

            if btype == "heading":
                flush()
                pending_heading = text
                pending_text = text + "\n"
                pending_btypes.add("heading")
                continue

            if _approx_tokens(pending_text) + toks > self.chunk_size and pending_text:
                flush()

            pending_text += text + "\n"
            pending_btypes.add(btype)

        flush()
        return self._merge_tiny(chunks)


    def _chunk_fixed_token(
        self,
        blocks: list[dict],
        doc_id: str,
        source_file: str,
        page_number: int,
    ) -> list[Chunk]:

        full_text = " ".join(
            b["text"].strip() for b in blocks if b["text"].strip()
        )
        heading = next(
            (b.get("section_heading") for b in blocks if b.get("section_heading")),
            None
        )
        btypes = sorted({b["block_type"] for b in blocks})
        return self._chunk_plain_text(
            full_text, doc_id, source_file, page_number, page_number,
            section_heading=heading, block_types=btypes,
        )

    def _chunk_plain_text(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        page_start: int,
        page_end: int,
        section_heading: Optional[str],
        block_types: list[str],
    ) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        window: list[str] = []
        tok = 0

        def emit(overlap_prefix: str = "", overlap_tok: int = 0):
            t = (overlap_prefix + " ".join(window)).strip()
            if t:
                chunks.append(Chunk(
                    chunk_id="",
                    doc_id=doc_id,
                    chunk_index=0,
                    total_chunks=0,
                    strategy=ChunkStrategy.FIXED_TOKEN,
                    text=t,
                    token_count=_approx_tokens(t),
                    source_file=source_file,
                    page_start=page_start,
                    page_end=page_end,
                    section_heading=section_heading,
                    block_types=block_types,
                    is_table=False,
                    overlap_tokens_prev=overlap_tok,
                ))

        for sent in sentences:
            st = _approx_tokens(sent)
            if tok + st > self.chunk_size and window:
                prefix, ptok = "", 0
                if chunks:
                    tail = chunks[-1].text.split(". ")
                    acc = ""
                    for ps in reversed(tail):
                        cand = ps + ". "
                        if _approx_tokens(acc + cand) <= self.overlap:
                            acc = cand + acc
                        else:
                            break
                    prefix, ptok = acc.strip() + " " if acc.strip() else "", _approx_tokens(acc)
                emit(prefix, ptok)
                window, tok = [sent], st
            else:
                window.append(sent)
                tok += st

        if window:
            emit()

        return self._merge_tiny(chunks)


    def _chunk_table(
        self,
        table: dict,
        doc_id: str,
        source_file: str,
        page_number: int,
    ) -> list[Chunk]:
        md = table.get("markdown", "")
        if not md.strip():
            return []

        toks = _approx_tokens(md)

        if toks <= self.table_max_tokens:
            return [Chunk(
                chunk_id="",
                doc_id=doc_id,
                chunk_index=0,
                total_chunks=0,
                strategy="table",
                text=md,
                token_count=toks,
                source_file=source_file,
                page_start=page_number,
                page_end=page_number,
                section_heading=table.get("caption"),
                block_types=["table"],
                is_table=True,
                extra={
                    "table_index": table.get("table_index", 0),
                    "extractor": table.get("extractor_used", ""),
                    "confidence": table.get("confidence"),
                },
            )]

        lines = md.split("\n")
        header = lines[:2] if len(lines) >= 2 else lines   
        body   = lines[2:] if len(lines) > 2 else []

        chunks: list[Chunk] = []
        current_rows: list[str] = []
        current_toks = _approx_tokens("\n".join(header))

        for row in body:
            rt = _approx_tokens(row)
            if current_toks + rt > self.table_max_tokens and current_rows:
                chunk_md = "\n".join(header + current_rows)
                chunks.append(self._make_table_chunk(
                    chunk_md, doc_id, source_file, page_number, table
                ))
                current_rows = [row]
                current_toks = _approx_tokens("\n".join(header)) + rt
            else:
                current_rows.append(row)
                current_toks += rt

        if current_rows:
            chunk_md = "\n".join(header + current_rows)
            chunks.append(self._make_table_chunk(
                chunk_md, doc_id, source_file, page_number, table
            ))

        return chunks

    def _make_table_chunk(
        self, md: str, doc_id: str, source_file: str, pn: int, table: dict
    ) -> Chunk:
        return Chunk(
            chunk_id="",
            doc_id=doc_id,
            chunk_index=0,
            total_chunks=0,
            strategy="table",
            text=md,
            token_count=_approx_tokens(md),
            source_file=source_file,
            page_start=pn,
            page_end=pn,
            section_heading=table.get("caption"),
            block_types=["table"],
            is_table=True,
            extra={
                "table_index": table.get("table_index", 0),
                "extractor": table.get("extractor_used", ""),
                "confidence": table.get("confidence"),
            },
        )


    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks
        result: list[Chunk] = [chunks[0]]
        for c in chunks[1:]:
            if c.token_count < self.min_chunk_size and not c.is_table and result:
                prev = result[-1]
                merged_text = prev.text + " " + c.text
                result[-1] = Chunk(
                    chunk_id="",
                    doc_id=prev.doc_id,
                    chunk_index=0,
                    total_chunks=0,
                    strategy=prev.strategy,
                    text=merged_text,
                    token_count=_approx_tokens(merged_text),
                    source_file=prev.source_file,
                    page_start=prev.page_start,
                    page_end=max(prev.page_end, c.page_end),
                    section_heading=prev.section_heading or c.section_heading,
                    block_types=sorted(set(prev.block_types) | set(c.block_types)),
                    is_table=False,
                    overlap_tokens_prev=prev.overlap_tokens_prev,
                )
            else:
                result.append(c)
        return result

    def _finalise(self, chunks: list[Chunk], doc_id: str) -> list[Chunk]:
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.chunk_index  = i
            c.total_chunks = total
            c.chunk_id     = Chunk.make_id(c.text, doc_id, i)
            if i > 0 and chunks[i].overlap_tokens_prev > 0:
                chunks[i - 1].overlap_tokens_next = chunks[i].overlap_tokens_prev
        return chunks
