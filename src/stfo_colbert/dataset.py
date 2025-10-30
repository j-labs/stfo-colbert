import io
import lzma
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymupdf
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from . import DELIMITER
from .utils import read_text_file, stream_text_file


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDataset:
    # Original input path (file or directory)
    source: Path
    # Factory function that returns an iterator over documents
    # This allows multiple passes without re-reading everything into RAM
    document_iterator_factory: Callable[[], Iterator[str]]
    # Optional document count if known upfront (used only for progress reporting)
    document_count: int | None = None

    def iter_documents(self) -> Iterator[str]:
        """Get an iterator over all documents."""
        return self.document_iterator_factory()


# -----------------
# Reading utilities
# -----------------


def _clean_delimiter(text: str) -> str:
    # Replace the delimiter if it appears inside a document to avoid splitting issues
    return text.replace(DELIMITER, "\n\n++++++++\n\n")


def _stream_documents_from_delimited_text(
    text_chunks: Iterator[str], delimiter: str
) -> Iterator[str]:
    """Stream documents from text chunks by splitting on delimiter.

    Handles delimiter that may span chunk boundaries.

    Args:
        text_chunks: Iterator yielding text chunks
        delimiter: Delimiter string to split on

    Yields:
        str: Individual documents
    """
    buffer = ""

    for chunk in text_chunks:
        buffer += chunk

        # Split on delimiter and yield complete documents
        parts = buffer.split(delimiter)

        # Keep last part in buffer (might be incomplete)
        buffer = parts[-1]

        # Yield all complete documents
        for part in parts[:-1]:
            stripped = part.strip()
            if stripped:
                yield stripped

    # Yield remaining buffer content
    if buffer.strip():
        yield buffer.strip()


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
    logger.info("Preparing dataset from delimited text file: %s", path)

    # Factory function to create a fresh iterator for each pass
    def document_iterator() -> Iterator[str]:
        logger.debug("Creating new document iterator for %s", path)
        return _stream_documents_from_delimited_text(stream_text_file(path), DELIMITER)

    return PreparedDataset(
        source=path,
        document_iterator_factory=document_iterator,
        document_count=None,  # Unknown without pre-counting
    )


def prepare_from_directory(dir_path: Path) -> PreparedDataset:
    logger.info("Preparing dataset from directory: %s", dir_path)

    # Check for existing compressed cache file
    cache_path = dir_path / ".stfo_colbert_cache.txt.xz"

    if cache_path.exists():
        logger.info("Found existing cache file: %s", cache_path)
        # Reuse existing compressed cache with streaming
        try:
            # Factory to stream from cache
            def document_iterator() -> Iterator[str]:
                logger.debug(
                    "Creating new document iterator from cache: %s", cache_path
                )
                with lzma.open(cache_path, "rt", encoding="utf-8") as f:
                    chunks = iter(lambda: f.read(8192), "")
                    yield from _stream_documents_from_delimited_text(chunks, DELIMITER)

            logger.info("Using cached dataset")
            return PreparedDataset(
                source=dir_path,
                document_iterator_factory=document_iterator,
                document_count=None,  # Unknown without pre-counting
            )
        except Exception as e:
            # If reading cache fails, fall through to re-parse
            logger.warning(
                "Failed reading cache at %s, will re-parse directory. Error: %s",
                cache_path,
                e,
            )

    # Parse directory contents and write cache incrementally
    logger.info("No cache found, parsing directory contents...")
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    logger.info("Found %d files to process", len(files))
    doc_count = 0

    # Write to cache as we process files
    try:
        logger.info("Writing cache incrementally to %s", cache_path)
        with lzma.open(cache_path, "wt", encoding="utf-8") as cache_file:
            for idx, p in enumerate(files, 1):
                suffix = p.suffix.lower()
                if suffix in (".txt", ".md"):
                    logger.debug(
                        "Processing text file %d/%d: %s", idx, len(files), p.name
                    )
                    text = _read_txt_or_md(p)
                elif suffix in (".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz"):
                    logger.debug(
                        "Processing document %d/%d: %s", idx, len(files), p.name
                    )
                    text = _read_document(p)
                else:
                    logger.debug("Ignoring unsupported file type: %s", p.name)
                    continue

                # Skip empty extractions
                text = text.strip()
                if text:
                    if doc_count > 0:
                        cache_file.write(DELIMITER)
                    cache_file.write(text)
                    doc_count += 1
                    if doc_count % 100 == 0:
                        logger.info("Processed %d documents so far...", doc_count)

        logger.info(
            "Successfully cached %d documents from %d files", doc_count, len(files)
        )
    except Exception as e:
        logger.exception("Failed writing cache to %s: %s", cache_path, e)
        # Continue anyway - we'll stream from files directly

    # Factory to stream from the cache we just created
    def document_iterator() -> Iterator[str]:
        logger.debug("Creating new document iterator from newly created cache")
        with lzma.open(cache_path, "rt", encoding="utf-8") as f:
            chunks = iter(lambda: f.read(8192), "")
            yield from _stream_documents_from_delimited_text(chunks, DELIMITER)

    logger.info(
        "Prepared dataset with %d documents from directory: %s", doc_count, dir_path
    )
    return PreparedDataset(
        source=dir_path,
        document_iterator_factory=document_iterator,
        document_count=doc_count,  # Known from cache creation
    )


def prepare_dataset(
    path: Path,
    model_name: str,
) -> PreparedDataset:
    path = path.expanduser().resolve()
    logger.info("Preparing dataset from path: %s", path)

    # First, prepare the raw dataset
    if path.is_dir():
        raw_dataset = prepare_from_directory(path)
    elif path.is_file():
        raw_dataset = prepare_from_delimited_txt(path)
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    # Create text splitter -> max length is taken from model config
    logger.info("Creating text splitter for model: %s", model_name)
    splitter = SentenceTransformersTokenTextSplitter(model_name=model_name)

    # Factory to stream chunks from raw documents
    def chunk_iterator() -> Iterator[str]:
        logger.debug("Creating new chunk iterator (will stream and split documents)")
        for doc in raw_dataset.iter_documents():
            chunks = splitter.split_text(doc)
            for chunk in chunks:
                yield chunk

    logger.info("Dataset prepared: ready for streaming with chunking")
    return PreparedDataset(
        source=raw_dataset.source,
        document_iterator_factory=chunk_iterator,
        document_count=None,  # We don't know chunk count without pre-counting
    )
