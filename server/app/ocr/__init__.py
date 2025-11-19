"""OCR processing module for utility meter reading recognition."""
from .processor import OCRProcessor
from .normalizer import normalize_ocr

__all__ = ["OCRProcessor", "normalize_ocr"]
