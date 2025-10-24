from dataclasses import dataclass
from pathlib import Path
import logging
import json
import lzma

from pylate import indexes, models, retrieve
from pylate.rank import RerankResult

from .utils import ensure_dir

logger = logging.getLogger(__name__)


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
    documents: list[str],
    document_ids: list[str],
    index_path: Path,
    model_name: str,
    batch_size: int = 32,
) -> IndexArtifacts:
    index_path = ensure_dir(index_path)

    model = load_model(model_name)

    # Reuse index directory if it exists; otherwise create a new one
    # Always override existing index to avoid accidental appends when re-indexing
    index = indexes.PLAID(
        index_folder=str(index_path),
        index_name="index",
        override=True,
    )

    # Always (re)encode and add documents when building the index
    logger.info("Encoding %d documents for indexing", len(documents))
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True,
    )
    logger.info("Adding encoded documents to index directory: %s", index_path)
    index.add_documents(
        documents_ids=document_ids,
        documents_embeddings=embeddings,
    )

    retriever = retrieve.ColBERT(index=index)
    logger.info("Index ready at %s", index_path)
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


def save_collection_json(collection: dict[str, str], index_path: Path) -> None:
    """Save collection mapping to compressed JSON file."""
    collection_file = index_path / "collection.json.xz"
    logger.info("Saving collection mapping to %s", collection_file)
    json_bytes = json.dumps(collection, ensure_ascii=False).encode("utf-8")
    with lzma.open(collection_file, "wb") as f:
        f.write(json_bytes)


def load_collection_json(index_path: Path) -> dict[str, str] | None:
    """Load collection mapping from compressed JSON file if it exists."""
    collection_file = index_path / "collection.json.xz"
    if not collection_file.exists():
        logger.warning(
            "No collection.json.xz found at %s - search results will not include text",
            index_path,
        )
        return None
    logger.info("Loading collection mapping from %s", collection_file)
    with lzma.open(collection_file, "rb") as f:
        json_bytes = f.read()
    return json.loads(json_bytes.decode("utf-8"))
