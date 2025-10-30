from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import logging
import sqlite3

from pylate import indexes, models, retrieve
from pylate.rank import RerankResult

from .utils import ensure_dir

logger = logging.getLogger(__name__)


class CollectionDB(Mapping):
    """Dict-like wrapper for SQLite collection storage with random access.

    Note: This class is not thread-safe. Each thread should create its own
    CollectionDB instance if concurrent access is needed.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self._ensure_table()

    def _ensure_table(self):
        """Create collection table if it doesn't exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collection (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def __getitem__(self, key: str) -> str:
        cursor = self.conn.execute(
            "SELECT text FROM collection WHERE doc_id = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __iter__(self):
        cursor = self.conn.execute("SELECT doc_id FROM collection")
        return (row[0] for row in cursor)

    def __len__(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM collection")
        return cursor.fetchone()[0]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def add(self, doc_id: str, text: str):
        """Add or update a document in the collection."""
        self.conn.execute(
            "INSERT OR REPLACE INTO collection (doc_id, text) VALUES (?, ?)",
            (doc_id, text),
        )

    def add_batch(self, items: list[tuple[str, str]]):
        """Add multiple documents efficiently."""
        logger.debug("Adding %d documents to collection.db", len(items))
        self.conn.executemany(
            "INSERT OR REPLACE INTO collection (doc_id, text) VALUES (?, ?)", items
        )

    def commit(self):
        """Commit pending changes."""
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        self.close()


class CollectionWriter:
    """Writer for incrementally saving collection documents per chunk."""

    def __init__(self, collection_file: Path, batch_size: int = 10000):
        """Initialize collection writer.

        Args:
            collection_file: Path to collection.db file
            batch_size: Number of documents to batch before committing
        """
        self.collection_file = collection_file
        self.batch_size = batch_size
        self.db = CollectionDB(collection_file)
        self.batch: list[tuple[str, str]] = []
        self.total_saved = 0

    def add_documents(self, documents: list[str], start_idx: int) -> None:
        """Add a chunk of documents to the collection.

        Args:
            documents: List of document texts
            start_idx: Starting document index for this chunk
        """
        for offset, doc in enumerate(documents):
            doc_id = str(start_idx + offset)
            self.batch.append((doc_id, doc))

            # Commit batch when full
            if len(self.batch) >= self.batch_size:
                try:
                    self.db.add_batch(self.batch)
                    self.db.commit()
                    self.total_saved += len(self.batch)
                    logger.info(
                        "Saved %d documents to collection (%d total saved)",
                        len(self.batch),
                        self.total_saved,
                    )
                    self.batch.clear()
                except Exception as e:
                    logger.error("Failed to save batch to collection: %s", e)
                    raise

    def finalize(self) -> int:
        """Flush remaining documents and return total count.

        Returns:
            Total number of documents saved
        """
        if self.batch:
            try:
                self.db.add_batch(self.batch)
                self.db.commit()
                self.total_saved += len(self.batch)
                logger.info(
                    "Saved final batch of %d documents to collection (%d total saved)",
                    len(self.batch),
                    self.total_saved,
                )
                self.batch.clear()
            except Exception as e:
                logger.error("Failed to save final batch to collection: %s", e)
                raise
        return self.total_saved

    def close(self):
        """Close the database connection."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()
        self.close()


@dataclass(frozen=True)
class IndexArtifacts:
    index_dir: Path
    model: models.ColBERT
    index: indexes.PLAID
    retriever: retrieve.ColBERT


def load_model(model_name: str) -> models.ColBERT:
    # PyLate will select the appropriate backend; we don't force device here.
    logger.info("Loading PyLate model: %s", model_name)
    return models.ColBERT(model_name_or_path=model_name)


def build_index(
    documents: Iterator[str],
    index_path: Path,
    model_name: str,
    batch_size: int = 64,
    encoding_chunk_size: int = 10000,
    document_count: int | None = None,
    collection_writer: CollectionWriter | None = None,
) -> IndexArtifacts:
    """Build index from streaming documents.

    Args:
        documents: Iterator yielding documents
        index_path: Path to store index
        model_name: Name of the model to use
        batch_size: Batch size for encoding (passed to model.encode)
        encoding_chunk_size: Number of documents to accumulate before encoding
            (reduces memory usage by processing in chunks)
        document_count: Optional total document count for progress reporting
        collection_writer: Optional CollectionWriter to save documents per chunk

    Returns:
        IndexArtifacts containing model, index, and retriever
    """
    index_path = ensure_dir(index_path)

    model = load_model(model_name)

    # Reuse index directory if it exists; otherwise create a new one
    # Always override existing index to avoid accidental appends when re-indexing
    index = indexes.PLAID(
        index_folder=str(index_path),
        index_name="index",
        override=True,
    )

    # Process documents in chunks to avoid loading all into RAM
    if document_count is not None:
        logger.info(
            "Starting streaming index build: %d total documents, chunk_size=%d, batch_size=%d",
            document_count,
            encoding_chunk_size,
            batch_size,
        )
    else:
        logger.info(
            "Starting streaming index build: chunk_size=%d, batch_size=%d",
            encoding_chunk_size,
            batch_size,
        )

    doc_chunk = []
    id_chunk = []
    total_processed = 0
    first_chunk = True
    chunk_number = 0

    try:
        for doc_idx, doc in enumerate(documents):
            doc_chunk.append(doc)
            id_chunk.append(str(doc_idx))

            # When chunk is full, encode and add to index
            if len(doc_chunk) >= encoding_chunk_size:
                chunk_number += 1
                if document_count is not None:
                    logger.info(
                        "Processing chunk %d: encoding documents %d-%d of %d (%.1f%% complete)",
                        chunk_number,
                        total_processed + 1,
                        total_processed + len(doc_chunk),
                        document_count,
                        (total_processed / document_count) * 100,
                    )
                else:
                    logger.info(
                        "Processing chunk %d: encoding documents %d-%d",
                        chunk_number,
                        total_processed + 1,
                        total_processed + len(doc_chunk),
                    )

                try:
                    embeddings = model.encode(
                        doc_chunk,
                        batch_size=batch_size,
                        is_query=False,
                        show_progress_bar=True,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to encode chunk %d (documents %d-%d): %s",
                        chunk_number,
                        total_processed + 1,
                        total_processed + len(doc_chunk),
                        e,
                    )
                    raise RuntimeError(
                        f"Encoding failed at chunk {chunk_number}. "
                        f"Index may be incomplete at {index_path}"
                    ) from e

                if first_chunk:
                    # First chunk: initialize index with override=True
                    logger.info(
                        "Initializing index at %s with first chunk. Clustering will happen - it takes time.",
                        index_path,
                    )
                    first_chunk = False
                else:
                    logger.info("Adding chunk %d to existing index", chunk_number)

                try:
                    index.add_documents(
                        documents_ids=id_chunk,
                        documents_embeddings=embeddings,
                    )
                except Exception as e:
                    logger.error("Failed to add chunk %d to index: %s", chunk_number, e)
                    raise RuntimeError(
                        f"Failed to add documents to index at chunk {chunk_number}. "
                        f"Index may be incomplete at {index_path}"
                    ) from e

                total_processed += len(doc_chunk)
                logger.info(
                    "Chunk %d indexed successfully (%d documents processed so far)",
                    chunk_number,
                    total_processed,
                )

                # Save this chunk to collection if writer is provided
                if collection_writer is not None:
                    chunk_start_idx = total_processed - len(doc_chunk)
                    collection_writer.add_documents(
                        doc_chunk, start_idx=chunk_start_idx
                    )

                doc_chunk.clear()
                id_chunk.clear()
    except Exception:
        logger.error(
            "Index build failed after processing %d documents. "
            "Cleaning up incomplete files at %s",
            total_processed,
            index_path,
        )
        # Note: We don't automatically delete the index directory here
        # as it may contain useful partial data for debugging
        raise

    # Handle remaining documents in the last chunk
    if doc_chunk:
        chunk_number += 1
        if document_count is not None:
            logger.info(
                "Processing final chunk %d: encoding documents %d-%d of %d (%.1f%% complete)",
                chunk_number,
                total_processed + 1,
                total_processed + len(doc_chunk),
                document_count,
                ((total_processed + len(doc_chunk)) / document_count) * 100,
            )
        else:
            logger.info(
                "Processing final chunk %d: encoding documents %d-%d",
                chunk_number,
                total_processed + 1,
                total_processed + len(doc_chunk),
            )

        try:
            embeddings = model.encode(
                doc_chunk,
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=True,
            )
        except Exception as e:
            logger.error(
                "Failed to encode final chunk %d (documents %d-%d): %s",
                chunk_number,
                total_processed + 1,
                total_processed + len(doc_chunk),
                e,
            )
            raise RuntimeError(
                f"Encoding failed at final chunk {chunk_number}. "
                f"Index may be incomplete at {index_path}"
            ) from e

        if first_chunk:
            logger.info("Initializing index at %s (single chunk only)", index_path)

        try:
            index.add_documents(
                documents_ids=id_chunk,
                documents_embeddings=embeddings,
            )
        except Exception as e:
            logger.error("Failed to add final chunk %d to index: %s", chunk_number, e)
            raise RuntimeError(
                f"Failed to add final documents to index at chunk {chunk_number}. "
                f"Index may be incomplete at {index_path}"
            ) from e

        total_processed += len(doc_chunk)
        logger.info(
            "Final chunk %d indexed successfully (%d total documents)",
            chunk_number,
            total_processed,
        )

        # Save final chunk to collection if writer is provided
        if collection_writer is not None:
            chunk_start_idx = total_processed - len(doc_chunk)
            collection_writer.add_documents(doc_chunk, start_idx=chunk_start_idx)

    retriever = retrieve.ColBERT(index=index)
    logger.info(
        "✓ Index build complete! Indexed %d documents in %d chunks at %s",
        total_processed,
        chunk_number,
        index_path,
    )

    # Validate collection count if writer was provided
    if collection_writer is not None:
        collection_count = collection_writer.total_saved
        if collection_count != total_processed:
            logger.warning(
                "Collection count mismatch: index has %d documents but collection has %d documents",
                total_processed,
                collection_count,
            )
        else:
            logger.info(
                "✓ Validation passed: collection and index both contain %d documents",
                total_processed,
            )

    return IndexArtifacts(
        index_dir=index_path, model=model, index=index, retriever=retriever
    )


def load_index_only(index_path: Path) -> retrieve.ColBERT:
    index = indexes.PLAID(
        index_folder=str(index_path),
        index_name="index",
        override=False,
    )
    return retrieve.ColBERT(index=index)


def encode_query(model: models.ColBERT, query: str):
    return model.encode([query], batch_size=1, is_query=True, show_progress_bar=False)


def retrieve_topk(
    retriever: retrieve.ColBERT, query_embeddings, k: int
) -> list[list[RerankResult]]:
    return retriever.retrieve(queries_embeddings=query_embeddings, k=k)


def save_collection_in_chunks(
    index_path: Path, batch_size: int = 10000
) -> CollectionWriter:
    """Create a CollectionWriter for incremental chunk-based saving.

    Args:
        index_path: Path to save collection file
        batch_size: Number of documents to batch before committing

    Returns:
        CollectionWriter instance (use as context manager)

    Example:
        with save_collection_in_chunks(index_path) as writer:
            writer.add_documents(chunk1, start_idx=0)
            writer.add_documents(chunk2, start_idx=len(chunk1))
    """
    collection_file = index_path / "collection.db"
    logger.info("Opening collection writer at %s", collection_file)
    return CollectionWriter(collection_file, batch_size=batch_size)


def load_collection(index_path: Path) -> CollectionDB | None:
    """Load collection mapping from SQLite database if it exists.

    Returns a dict-like CollectionDB object for random access without loading
    the entire collection into memory.
    """
    collection_file = index_path / "collection.db"
    if not collection_file.exists():
        logger.warning(
            "No collection.db found at %s - search results will not include text",
            index_path,
        )
        return None
    logger.info("Loading collection mapping from %s", collection_file)
    return CollectionDB(collection_file)
