"""Text extraction helpers for recruiter-uploaded resume files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractionResult:
    """The extracted text and a user-safe status message."""

    text: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.text) and self.error is None


def _read_uploaded_bytes(uploaded_file: Any) -> bytes:
    """Read bytes from an UploadedFile-like object and restore its cursor."""
    if not hasattr(uploaded_file, "read"):
        raise TypeError("The supplied object does not provide a read() method.")

    data = uploaded_file.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    if not isinstance(data, bytes):
        return bytes(data)
    return data


def extract_pdf_text(file_bytes: bytes) -> ExtractionResult:
    """Extract text from a PDF byte string without exposing parser errors."""
    if not file_bytes:
        return ExtractionResult("", "The PDF file is empty.")

    try:
        import fitz
    except ImportError:
        return ExtractionResult(
            "",
            "PDF support is unavailable because PyMuPDF is not installed.",
        )

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except (fitz.FileDataError, ValueError, RuntimeError):
        return ExtractionResult("", "The PDF file is invalid or could not be opened.")

    try:
        pages = [page.get_text("text") for page in document]
    finally:
        document.close()

    text = "\n".join(pages).strip()
    if not text:
        return ExtractionResult(
            "",
            "No selectable text was found. This PDF may be scanned and require OCR.",
        )
    return ExtractionResult(text)


def extract_txt_text(file_bytes: bytes) -> ExtractionResult:
    """Decode TXT bytes using common encodings, with a safe fallback."""
    if not file_bytes:
        return ExtractionResult("", "The TXT file is empty.")

    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            text = file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
        if text:
            return ExtractionResult(text)
        return ExtractionResult("", "The TXT file contains no readable text.")

    return ExtractionResult("", "The TXT file encoding could not be read.")


def extract_text_from_file(uploaded_file: Any) -> ExtractionResult:
    """Extract text from a Streamlit UploadedFile or similar file object."""
    filename = getattr(uploaded_file, "name", "")
    suffix = Path(str(filename)).suffix.lower()

    if suffix not in {".pdf", ".txt"}:
        return ExtractionResult("", "Unsupported file type. Upload a PDF or TXT file.")

    try:
        file_bytes = _read_uploaded_bytes(uploaded_file)
    except (OSError, TypeError, ValueError):
        return ExtractionResult("", "The uploaded file could not be read.")

    if suffix == ".pdf":
        return extract_pdf_text(file_bytes)
    return extract_txt_text(file_bytes)
