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
    """Dict-like wrapper for SQLite collection storage with random access."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
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
        logger.info("Adding %d documents to collection.db", len(items))
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
            embeddings = model.encode(
                doc_chunk,
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=True,
            )

            if first_chunk:
                # First chunk: initialize index with override=True
                logger.info(
                    "Initializing index at %s with first chunk. Clustering will happen - it takes time.",
                    index_path,
                )
                first_chunk = False
            else:
                logger.info("Adding chunk %d to existing index", chunk_number)

            index.add_documents(
                documents_ids=id_chunk,
                documents_embeddings=embeddings,
            )

            total_processed += len(doc_chunk)
            logger.info(
                "Chunk %d indexed successfully (%d documents processed so far)",
                chunk_number,
                total_processed,
            )
            doc_chunk.clear()
            id_chunk.clear()

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
        embeddings = model.encode(
            doc_chunk,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )

        if first_chunk:
            logger.info("Initializing index at %s (single chunk only)", index_path)

        index.add_documents(
            documents_ids=id_chunk,
            documents_embeddings=embeddings,
        )
        total_processed += len(doc_chunk)
        logger.info(
            "Final chunk %d indexed successfully (%d total documents)",
            chunk_number,
            total_processed,
        )

    retriever = retrieve.ColBERT(index=index)
    logger.info(
        "✓ Index build complete! Indexed %d documents in %d chunks at %s",
        total_processed,
        chunk_number,
        index_path,
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


def save_collection(collection: dict[str, str], index_path: Path) -> None:
    """Save collection mapping to SQLite database."""
    collection_file = index_path / "collection.db"
    logger.info("Saving collection mapping to %s", collection_file)

    with CollectionDB(collection_file) as db:
        # Batch insert for efficiency
        items = list(collection.items())
        batch_size = 10000
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            db.add_batch(batch)
            if (i + batch_size) % 50000 == 0:
                logger.info("Saved %d documents...", i + batch_size)
        db.commit()

    # Log file size
    db_size = collection_file.stat().st_size
    logger.info("✓ Collection saved: %.2f MB", db_size / (1024 * 1024))


def save_collection_streaming(
    documents: Iterator[str], index_path: Path, document_count: int | None = None
) -> None:
    """Save collection mapping to SQLite database using streaming.

    Args:
        documents: Iterator yielding documents
        index_path: Path to save collection file
        document_count: Optional total document count for progress reporting
    """
    collection_file = index_path / "collection.db"
    logger.info("Saving collection mapping to %s (streaming)", collection_file)
    if document_count is not None:
        logger.info(
            "Streaming through %d documents to build collection map...", document_count
        )

    with CollectionDB(collection_file) as db:
        batch = []
        batch_size = 10000
        doc_count = 0

        for doc_idx, doc in enumerate(documents):
            batch.append((str(doc_idx), doc))
            doc_count += 1

            # Commit batch when full
            if len(batch) >= batch_size:
                db.add_batch(batch)
                db.commit()
                batch.clear()

                if doc_count % 50000 == 0:
                    if document_count is not None:
                        logger.info(
                            "Saved %d/%d documents...",
                            doc_count,
                            document_count,
                        )
                    else:
                        logger.info("Saved %d documents...", doc_count)

        # Save remaining documents
        if batch:
            db.add_batch(batch)
            db.commit()

        logger.info("Built collection database with %d documents", doc_count)

    # Log file size
    db_size = collection_file.stat().st_size
    logger.info("✓ Collection saved: %.2f MB", db_size / (1024 * 1024))


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
