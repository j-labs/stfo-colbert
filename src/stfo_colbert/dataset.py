import io
import lzma
import logging
from dataclasses import dataclass
from pathlib import Path

from . import DELIMITER
from .utils import read_text_file


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDataset:
    # Original input path (file or directory)
    source: Path
    # Document IDs and their content (kept in-memory for serving text in responses)
    document_ids: list[str]
    documents: list[str]


# -----------------
# Reading utilities
# -----------------


def _clean_delimiter(text: str) -> str:
    # Replace the delimiter if it appears inside a document to avoid splitting issues
    return text.replace(DELIMITER, "\n\n++++++++\n\n")


def _read_txt_or_md(path: Path) -> str:
    return _clean_delimiter(read_text_file(path))


def _read_document(path: Path) -> str:
    """Read document content from various formats (PDF, XPS, EPUB, MOBI, FB2, CBZ, SVG)."""
    try:
        import pymupdf
    except Exception as e:
        logger.exception("Failed to import pymupdf for reading %s: %s", path, e)
        return ""
    try:
        doc = pymupdf.open(str(path))
        out = io.StringIO()
        for page in doc:
            out.write(page.get_text() or "")
            out.write("\n")
        doc.close()
        return _clean_delimiter(out.getvalue())
    except Exception as e:
        logger.exception("Failed to extract text from %s: %s", path, e)
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
        except Exception as e:
            # If reading cache fails, fall through to re-parse
            logger.exception("Failed reading cache at %s, will re-parse directory. Error: %s", cache_path, e)

    # Parse directory contents
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    docs: list[str] = []
    for p in files:
        suffix = p.suffix.lower()
        text = ""
        if suffix in (".txt", ".md"):
            text = _read_txt_or_md(p)
        elif suffix in (".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz", ".svg"):
            text = _read_document(p)
        # Skip empty extractions
        if text.strip():
            docs.append(text.strip())

    ids = [str(i) for i in range(len(docs))]

    # Create cache file directly in the source directory
    try:
        with lzma.open(cache_path, 'wt', encoding='utf-8') as f:
            f.write(DELIMITER.join(docs))
    except Exception as e:
        logger.exception("Failed writing cache to %s: %s", cache_path, e)

    return PreparedDataset(source=dir_path, document_ids=ids, documents=docs)


def prepare_dataset(path: Path) -> PreparedDataset:
    path = path.expanduser().resolve()
    if path.is_dir():
        return prepare_from_directory(path)
    if path.is_file():
        return prepare_from_delimited_txt(path)
    raise FileNotFoundError(f"Dataset path does not exist: {path}")
