import argparse
import logging
from pathlib import Path

from .dataset import prepare_dataset
from .indexer import IndexArtifacts, build_index, load_index_only, save_collection_json, load_collection_json
from .server import create_app, run_server

DEFAULT_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-17m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="stfo-colbert: straightforward ColBERT indexing and serving")
    p.add_argument("--port", type=int, default=8889, help="Port to serve on (default: 8889)")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="Hugging Face model id/name")
    p.add_argument("--index-path", type=Path, help="Path to an existing PyLate index directory")
    p.add_argument("--dataset-path", type=Path, help="Path to dataset for index creation (file or directory)")
    return p.parse_args()


def main() -> None:
    # Configure basic logging for CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("stfo_colbert.cli")

    args = parse_args()

    if bool(args.index_path) == bool(args.dataset_path):
        raise SystemExit("Provide exactly one of --index-path or --dataset-path")

    if args.index_path:
        logger.info("Loading existing index from %s", args.index_path)
        try:
            retriever = load_index_only(args.index_path)
            # Load a model for query encoding even in index-only mode
            from .indexer import load_model
            model = load_model(args.model_name)
            # Try to load collection mapping if available
            collection = load_collection_json(args.index_path)
            app = create_app(model=model, retriever=retriever, collection=collection)
            run_server(app, port=args.port)
            return
        except Exception as e:
            logger.exception("CLI failed in index-only mode: %s", e)
            raise SystemExit(1)

    # Dataset path provided: build index then serve
    logger.info("Preparing dataset at %s", args.dataset_path)
    prepared = prepare_dataset(args.dataset_path, args.model_name)
    try:
        # By default create or reuse an index directory next to the dataset
        index_dir = (Path.cwd() / "stfo_indexes" / args.dataset_path.stem)
        artifacts: IndexArtifacts = build_index(
            documents=prepared.documents,
            document_ids=prepared.document_ids,
            index_path=index_dir,
            model_name=args.model_name,
            batch_size=32,
        )

        # Prepare a collection map for returning text fields
        collection_map = {pid: text for pid, text in zip(prepared.document_ids, prepared.documents)}

        # Save collection mapping alongside the index
        save_collection_json(collection_map, index_dir)

        app = create_app(model=artifacts.model, retriever=artifacts.retriever, collection=collection_map)
        run_server(app, port=args.port)
    except Exception as e:
        logger.exception("CLI failed while building/serving from dataset: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
