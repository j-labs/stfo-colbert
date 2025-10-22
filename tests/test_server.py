from __future__ import annotations

from fastapi.testclient import TestClient

from stfo_colbert.server import create_app


def test_health_and_search(monkeypatch):
    # Monkeypatch encode/retrieve to avoid loading heavy model
    from stfo_colbert import server as srv

    def fake_encode_query(model, q):
        return [0.0]

    def fake_retrieve_topk(retriever, q_emb, k):
        return [[("0", 1.0), ("1", 0.5)]]

    monkeypatch.setattr(srv, "encode_query", fake_encode_query)
    monkeypatch.setattr(srv, "retrieve_topk", fake_retrieve_topk)

    collection = {"0": "Zero", "1": "One"}
    app = create_app(model=None, retriever=None, collection=collection)
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/search", params={"query": "hello", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "hello"
    assert len(data["topk"]) == 2
    # probs should sum to ~1
    s = sum(item.get("prob", 0.0) for item in data["topk"])
    assert 0.99 <= s <= 1.01
