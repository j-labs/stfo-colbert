import logging
import math
from functools import lru_cache
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pylate import models, retrieve

from .indexer import encode_query, retrieve_topk

logger = logging.getLogger(__name__)


def _results_to_topk(results: Any, collection: Mapping[str, str] | None) -> list[dict[str, Any]]:
    topk: list[dict[str, Any]] = []
    all_scores: list[float] = []

    # PyLate might return: list[[ (id, score), ... ]] OR list[[ {id:..., score:...}, ... ]] OR a DataFrame
    if isinstance(results, list):
        if len(results) > 0:
            inner = results[0]
            if hasattr(inner, "iterrows"):
                for rank, (_, row) in enumerate(inner.iterrows()):
                    doc_id = str(row["id"]) if "id" in row else str(row.get("pid", rank))
                    score = float(row["score"]) if "score" in row else float(row.get("score", 0.0))
                    text = collection.get(doc_id) if collection else None
                    d = {"pid": doc_id, "rank": rank, "score": score}
                    if text is not None:
                        d["text"] = text
                    topk.append(d)
                    all_scores.append(score)
            elif isinstance(inner, list):
                for rank, item in enumerate(inner):
                    if isinstance(item, dict):
                        doc_id = str(item.get("id", item.get("pid", rank)))
                        score = float(item.get("score", 0.0))
                    else:
                        # tuple/list (id, score)
                        try:
                            doc_id = str(item[0])
                            score = float(item[1])
                        except Exception:
                            continue
                    text = collection.get(doc_id) if collection else None
                    d = {"pid": doc_id, "rank": rank, "score": score}
                    if text is not None:
                        d["text"] = text
                    topk.append(d)
                    all_scores.append(score)
    elif hasattr(results, "iterrows"):
        for rank, (_, row) in enumerate(results.iterrows()):
            doc_id = str(row["id"]) if "id" in row else str(row.get("pid", rank))
            score = float(row["score"]) if "score" in row else float(row.get("score", 0.0))
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
