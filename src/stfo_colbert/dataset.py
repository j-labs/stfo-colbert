import io
import lzma
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

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
        doc = pymupdf.open(str(path))
        out = io.StringIO()
        for i, page in enumerate(doc.pages()):
            out.write(f"---- Beginning of page {i + 1} ----\n\n")
            out.write(page.get_text() or "")
            out.write(f"\n\n---- End of page {i + 1} ----\n\n")
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
            with lzma.open(cache_path, "rt", encoding="utf-8") as f:
                raw = f.read()
            parts = [p.strip() for p in raw.split(DELIMITER)]
            docs = [p for p in parts if p]
            ids = [str(i) for i in range(len(docs))]
            return PreparedDataset(source=dir_path, document_ids=ids, documents=docs)
        except Exception as e:
            # If reading cache fails, fall through to re-parse
            logger.exception(
                "Failed reading cache at %s, will re-parse directory. Error: %s",
                cache_path,
                e,
            )

    # Parse directory contents
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    docs: list[str] = []
    for p in files:
        suffix = p.suffix.lower()
        if suffix in (".txt", ".md"):
            text = _read_txt_or_md(p)
        elif suffix in (".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz"):
            text = _read_document(p)
        else:
            logger.warning("Ignoring unsupported file type: %s", suffix)
            continue
        # Skip empty extractions
        if text.strip():
            docs.append(
                text.strip()
            )  # TODO: instead of doing it in RAM, dump to disk and stream from there

    ids = [str(i) for i in range(len(docs))]

    # Create cache file directly in the source directory
    try:
        with lzma.open(cache_path, "wt", encoding="utf-8") as f:
            f.write(DELIMITER.join(docs))
    except Exception as e:
        logger.exception("Failed writing cache to %s: %s", cache_path, e)

    return PreparedDataset(source=dir_path, document_ids=ids, documents=docs)


def prepare_dataset(
    path: Path,
    model_name: str,
) -> PreparedDataset:
    path = path.expanduser().resolve()

    # First, prepare the raw dataset
    if path.is_dir():
        raw_dataset = prepare_from_directory(path)
    elif path.is_file():
        raw_dataset = prepare_from_delimited_txt(path)
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    # Create text splitter -> max length is taken from model config
    splitter = SentenceTransformersTokenTextSplitter(model_name=model_name)

    # Split documents into chunks
    all_chunks: list[str] = []
    for doc in raw_dataset.documents:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)

    # Generate IDs for chunks
    chunk_ids = [str(i) for i in range(len(all_chunks))]

    return PreparedDataset(
        source=raw_dataset.source,
        document_ids=chunk_ids,
        documents=all_chunks,
    )
