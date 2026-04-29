from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import fitz  
from PIL import Image

logger = logging.getLogger(__name__)


class OCRBackend(str, Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"


@dataclass
class OCRResult:
    page_number: int
    text: str
    backend_used: str
    confidence: float | None  


class OCRHandler:


    def __init__(
        self,
        backend: OCRBackend = OCRBackend.TESSERACT,
        dpi: int = 300,
        language: str = "eng",
    ):
        self.backend = backend
        self.dpi = dpi
        self.language = language
        self._easyocr_reader = None   

    def ocr_page(self, pdf_path: str | Path, page_number: int) -> OCRResult:
        """OCR a single page (1-indexed)."""
        img = self._render_page(str(pdf_path), page_number)
        return self._run_ocr(img, page_number)

    def ocr_pages(
        self, pdf_path: str | Path, page_numbers: list[int]
    ) -> list[OCRResult]:
        """OCR multiple pages."""
        results = []
        path = str(pdf_path)
        for pn in page_numbers:
            try:
                img = self._render_page(path, pn)
                results.append(self._run_ocr(img, pn))
            except Exception as e:
                logger.error("OCR failed on page %d: %s", pn, e)
        return results


    def _render_page(self, pdf_path: str, page_number: int) -> Image.Image:
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img


    def _run_ocr(self, img: Image.Image, page_number: int) -> OCRResult:
        if self.backend == OCRBackend.TESSERACT:
            return self._tesseract(img, page_number)
        return self._easyocr(img, page_number)

    def _tesseract(self, img: Image.Image, page_number: int) -> OCRResult:
        import pytesseract

        data = pytesseract.image_to_data(
            img,
            lang=self.language,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        words, confs = [], []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if word:
                words.append(word)
                c = data["conf"][i]
                if isinstance(c, (int, float)) and c >= 0:
                    confs.append(float(c))

        text = " ".join(words)
        avg_conf = sum(confs) / len(confs) if confs else None
        return OCRResult(
            page_number=page_number,
            text=text,
            backend_used="tesseract",
            confidence=round(avg_conf, 1) if avg_conf else None,
        )

    def _easyocr(self, img: Image.Image, page_number: int) -> OCRResult:
        import easyocr
        import numpy as np

        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(
                [self.language], gpu=False, verbose=False
            )

        img_array = np.array(img)
        results = self._easyocr_reader.readtext(img_array, detail=1)
        texts = [r[1] for r in results if r[2] > 0.3]   
        confs = [r[2] for r in results if r[2] > 0.3]

        text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) * 100 if confs else None
        return OCRResult(
            page_number=page_number,
            text=text,
            backend_used="easyocr",
            confidence=round(avg_conf, 1) if avg_conf else None,
        )
