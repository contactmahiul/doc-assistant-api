
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)



@dataclass
class ExtractedTable:
    page_number: int          
    table_index: int           
    extractor_used: str        
    dataframe: pd.DataFrame
    markdown: str             
    bbox: Optional[tuple[float, float, float, float]] = None   
    confidence: Optional[float] = None   
    caption: Optional[str] = None         



def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    df = df.fillna("").astype(str)
    if _looks_like_header(df.iloc[0].tolist()):
        headers = df.iloc[0].tolist()
        rows = df.iloc[1:].values.tolist()
    else:
        headers = [f"col_{i}" for i in range(len(df.columns))]
        rows = df.values.tolist()

    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list) -> str:
        return "| " + " | ".join(
            str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
        ) + " |"

    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(headers), separator] + [fmt_row(row) for row in rows]
    return "\n".join(lines)


def _looks_like_header(row: list[str]) -> bool:
    if not row:
        return False
    non_numeric = sum(1 for cell in row if cell.strip() and not _is_numeric(cell))
    return non_numeric >= len(row) * 0.5


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("%", "").strip())
        return True
    except ValueError:
        return False


def _clean_cell(val) -> str:
    if val is None:
        return ""
    return str(val).strip().replace("\n", " ")



class TableExtractor:


    def __init__(
        self,
        min_rows: int = 2,
        min_cols: int = 2,
        confidence_threshold: float = 50.0,
        use_camelot_fallback: bool = True,
    ):
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.confidence_threshold = confidence_threshold
        self.use_camelot_fallback = use_camelot_fallback


    def extract(self, pdf_path: str | Path) -> list[ExtractedTable]:
        path = Path(pdf_path).resolve()
        results: list[ExtractedTable] = []

        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_number = page_idx + 1
                page_tables = self._extract_with_pdfplumber(page, page_number)

                if not page_tables and self.use_camelot_fallback:
                    page_tables = self._extract_with_camelot(
                        str(path), page_number
                    )

                results.extend(page_tables)

        return results

    def extract_page(
        self, pdf_path: str | Path, page_number: int
    ) -> list[ExtractedTable]:
        path = Path(pdf_path).resolve()

        with pdfplumber.open(str(path)) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                raise ValueError(f"Page {page_number} out of range")
            page = pdf.pages[page_number - 1]
            tables = self._extract_with_pdfplumber(page, page_number)

        if not tables and self.use_camelot_fallback:
            tables = self._extract_with_camelot(str(path), page_number)

        return tables


    def _extract_with_pdfplumber(
        self, page: pdfplumber.page.Page, page_number: int
    ) -> list[ExtractedTable]:
        results: list[ExtractedTable] = []

        try:
            raw_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                }
            )
        except Exception as e:
            logger.warning("pdfplumber failed on page %d: %s", page_number, e)
            return results

        for idx, raw in enumerate(raw_tables):
            if not raw:
                continue
            cleaned = [
                [_clean_cell(cell) for cell in row] for row in raw
            ]
            df = pd.DataFrame(cleaned)

            if df.shape[0] < self.min_rows or df.shape[1] < self.min_cols:
                continue

            bbox = None
            try:
                tboxes = page.find_tables()
                if idx < len(tboxes):
                    b = tboxes[idx].bbox
                    bbox = (b[0], b[1], b[2], b[3])
            except Exception:
                pass

            results.append(
                ExtractedTable(
                    page_number=page_number,
                    table_index=idx,
                    extractor_used="pdfplumber",
                    dataframe=df,
                    markdown=_df_to_markdown(df),
                    bbox=bbox,
                )
            )

        return results


    def _extract_with_camelot(
        self, pdf_path: str, page_number: int
    ) -> list[ExtractedTable]:
        try:
            import camelot
        except ImportError:
            logger.warning("camelot not installed; skipping fallback")
            return []

        results: list[ExtractedTable] = []
        page_str = str(page_number)

        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=page_str,
                    flavor=flavor,
                    suppress_stdout=True,
                )
            except Exception as e:
                logger.debug("camelot %s failed on page %d: %s", flavor, page_number, e)
                continue

            for idx, table in enumerate(tables):
                if table.accuracy < self.confidence_threshold:
                    continue
                df = table.df.copy()
                if df.shape[0] < self.min_rows or df.shape[1] < self.min_cols:
                    continue

                results.append(
                    ExtractedTable(
                        page_number=page_number,
                        table_index=idx,
                        extractor_used=f"camelot-{flavor}",
                        dataframe=df,
                        markdown=_df_to_markdown(df),
                        bbox=tuple(table._bbox) if hasattr(table, "_bbox") else None,
                        confidence=round(table.accuracy, 1),
                    )
                )

            if results:
                break   

        return results
