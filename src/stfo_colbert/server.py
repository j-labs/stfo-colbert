import logging
import math
from functools import lru_cache
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pylate import models, retrieve
from pylate.rank import RerankResult

from .indexer import encode_query, retrieve_topk

logger = logging.getLogger(__name__)


def _results_to_topk(results: list[list[RerankResult]], collection: Mapping[str, str] | None) -> list[dict[str, Any]]:
    topk: list[dict[str, Any]] = []
    all_scores: list[float] = []

    # Process list[list[RerankResult]] (RerankResult is a typed dict!)
    if results:
        for rank, result in enumerate(results[0]):
            doc_id = result["id"]
            score = result["score"]
            text = collection.get(doc_id) if collection else None
            d = {"pid": doc_id, "rank": rank, "score": score}
            if text is not None:
                d["text"] = text
            topk.append(d)
            all_scores.append(score)

    # Probabilities via softmax over scores
    if all_scores:
        probs = [math.exp(s) for s in all_scores]
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)
        for item, prob in zip(topk, probs):
            item["prob"] = prob

    topk.sort(key=lambda x: (-x["score"], x["pid"]))
    return topk


def create_app(model: models.ColBERT, retriever: retrieve.ColBERT, collection: Mapping[str, str] | None = None) -> FastAPI:
    app = FastAPI()

    @lru_cache(maxsize=1024)
    def cached_query(q: str, k: int) -> dict[str, Any]:
        q_emb = encode_query(model, q)
        results = retrieve_topk(retriever, q_emb, k)
        topk = _results_to_topk(results, collection)
        return {"query": q, "topk": topk}

    @app.get("/search")
    def search(query: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=100)):
        try:
            return cached_query(query, k)
        except Exception as e:
            logger.exception("Search failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(app: FastAPI, port: int) -> None:
    import uvicorn

    logger.info("[stfo-colbert] Serving on http://0.0.0.0:%d", port)
    logger.info("[stfo-colbert] Endpoints: /search?query=...&k=5")
    uvicorn.run(app, host="0.0.0.0", port=port)
