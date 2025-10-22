import io
import lzma
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import DELIMITER
from .utils import read_text_file, write_text_file


@dataclass(frozen=True)
class PreparedDataset:
    # Original input path (file or directory)
    source: Path
    # Document IDs and their content (kept in-memory for serving text in responses)
    document_ids: list[str]
    documents: list[str]

    def cleanup(self) -> None:
        # No cleanup needed - cache files are persisted
        pass


# -----------------
# Reading utilities
# -----------------


def _clean_delimiter(text: str) -> str:
    # Replace the delimiter if it appears inside a document to avoid splitting issues
    return text.replace(DELIMITER, "\n\n++++++++\n\n")


def _read_txt_or_md(path: Path) -> str:
    return _clean_delimiter(read_text_file(path))


def _read_pdf_if_available(path: Path) -> str:
    try:
        from pypdf import PdfReader  # optional
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        out = io.StringIO()
        for page in reader.pages:
            out.write(page.extract_text() or "")
            out.write("\n")
        return _clean_delimiter(out.getvalue())
    except Exception:
        return ""


def _read_docx_if_available(path: Path) -> str:
    try:
        import docx  # type: ignore  # optional python-docx
    except Exception:
        return ""
    try:
        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        return _clean_delimiter("\n".join(parts))
    except Exception:
        return ""


def _read_odt_if_available(path: Path) -> str:
    try:
        from odf.opendocument import load  # optional
        from odf.text import P
    except Exception:
        return ""
    try:
        txts: list[str] = []
        doc = load(str(path))
        for p in doc.getElementsByType(P):
            if p.firstChild is not None:
                txts.append(str(p.firstChild.data))
        return _clean_delimiter("\n".join(txts))
    except Exception:
        return ""


# -----------------
# Public API
# -----------------


def prepare_from_delimited_txt(path: Path) -> PreparedDataset:
    raw = read_text_file(path)
    parts = [p.strip() for p in raw.split(DELIMITER)]
    docs = [p for p in parts if p]
    ids = [str(i) for i in range(len(docs))]
    return PreparedDataset(source=path, document_ids=ids, documents=docs)


def prepare_from_directory(dir_path: Path) -> PreparedDataset:
    # Check for existing compressed cache file
    cache_path = dir_path / ".stfo_colbert_cache.txt.xz"

    if cache_path.exists():
        # Reuse existing compressed cache
        try:
            with lzma.open(cache_path, 'rt', encoding='utf-8') as f:
                raw = f.read()
            parts = [p.strip() for p in raw.split(DELIMITER)]
            docs = [p for p in parts if p]
            ids = [str(i) for i in range(len(docs))]
            return PreparedDataset(source=dir_path, document_ids=ids, documents=docs)
        except Exception:
            # If reading cache fails, fall through to re-parse
            pass

    # Parse directory contents
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    docs: list[str] = []
    for p in files:
        suffix = p.suffix.lower()
        text = ""
        if suffix in {".txt", ".md"}:
            text = _read_txt_or_md(p)
        elif suffix == ".pdf":
            text = _read_pdf_if_available(p)
        elif suffix == ".docx":
            text = _read_docx_if_available(p)
        elif suffix == ".odt":
            text = _read_odt_if_available(p)
        # Skip empty extractions
        if text.strip():
            docs.append(text.strip())

    ids = [str(i) for i in range(len(docs))]

    # Create cache file directly in the source directory
    try:
        with lzma.open(cache_path, 'wt', encoding='utf-8') as f:
            f.write(DELIMITER.join(docs))
    except Exception:
        pass

    return PreparedDataset(source=dir_path, document_ids=ids, documents=docs)


def prepare_dataset(path: Path) -> PreparedDataset:
    path = path.expanduser().resolve()
    if path.is_dir():
        return prepare_from_directory(path)
    if path.is_file():
        return prepare_from_delimited_txt(path)
    raise FileNotFoundError(f"Dataset path does not exist: {path}")
