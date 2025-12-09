# Core/document.py
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl

def _is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")

def extract_text_from_bytes(data: bytes, filename: str = "") -> str:
    name = (filename or "").lower()

    # PDF
    if name.endswith(".pdf") or _is_pdf(data):
        if not data:
            return ""
        doc = fitz.open(stream=data, filetype="pdf")
        return "".join(p.get_text() for p in doc)

    # DOCX
    if name.endswith(".docx"):
        if not data:
            return ""
        d = DocxDocument(BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    # TXT
    if name.endswith(".txt"):
        return (data or b"").decode("utf-8", errors="ignore")

    # PPTX
    if name.endswith(".pptx"):
        if not data:
            return ""
        prs = Presentation(BytesIO(data))
        out = []
        for s in prs.slides:
            for sh in s.shapes:
                if hasattr(sh, "text"):
                    out.append(sh.text)
        return "\n".join(out)

    # XLSX
    if name.endswith(".xlsx"):
        if not data:
            return ""
        wb = openpyxl.load_workbook(BytesIO(data), data_only=True)
        lines = []
        for sh in wb.worksheets:
            for row in sh.iter_rows(values_only=True):
                cells = [str(c) for c in row if c not in (None, "")]
                if cells:
                    lines.append(" ".join(cells))
        return "\n".join(lines)

    return ""

def extract_text(file) -> str:
    """Compat rétro: lit le flux entier et appelle extract_text_from_bytes."""
    if not file:
        return ""
    try:
        pos = getattr(file, "tell", lambda: 0)()
        data = file.read() or b""
        # si possible, revenir au début pour ne pas casser l'appelant
        try:
            file.seek(0)
        except Exception:
            pass
        filename = getattr(file, "name", "") or ""
        return extract_text_from_bytes(data, filename)
    except Exception:
        return ""



def chunk_text(text, size=1000, overlap=200):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)]
