from fastapi.testclient import TestClient

from stfo_colbert.server import create_app


def test_search_endpoint(monkeypatch):
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

    r = client.get("/search", params={"query": "hello", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "hello"
    assert len(data["topk"]) == 2
    # probs should sum to ~1
    s = sum(item.get("prob", 0.0) for item in data["topk"])
    assert 0.99 <= s <= 1.01


def test_search_missing_query():
    app = create_app(model=None, retriever=None, collection={})
    client = TestClient(app)

    # Missing query parameter
    r = client.get("/search")
    assert r.status_code == 422  # Validation error


def test_search_with_text_in_results(monkeypatch):
    from stfo_colbert import server as srv

    def fake_encode_query(model, q):
        return [0.0]

    def fake_retrieve_topk(retriever, q_emb, k):
        return [[("doc1", 2.5), ("doc2", 1.3)]]

    monkeypatch.setattr(srv, "encode_query", fake_encode_query)
    monkeypatch.setattr(srv, "retrieve_topk", fake_retrieve_topk)

    collection = {"doc1": "First document", "doc2": "Second document"}
    app = create_app(model=None, retriever=None, collection=collection)
    client = TestClient(app)

    r = client.get("/search", params={"query": "test", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert len(data["topk"]) == 2
    # Verify text is included
    assert data["topk"][0]["text"] == "First document"
    assert data["topk"][1]["text"] == "Second document"
    assert data["topk"][0]["pid"] == "doc1"
    assert data["topk"][1]["pid"] == "doc2"


def test_search_caching(monkeypatch):
    from stfo_colbert import server as srv

    call_count = [0]

    def fake_encode_query(model, q):
        call_count[0] += 1
        return [0.0]

    def fake_retrieve_topk(retriever, q_emb, k):
        return [[("0", 1.0)]]

    monkeypatch.setattr(srv, "encode_query", fake_encode_query)
    monkeypatch.setattr(srv, "retrieve_topk", fake_retrieve_topk)

    app = create_app(model=None, retriever=None, collection={"0": "Doc"})
    client = TestClient(app)

    # First request
    r1 = client.get("/search", params={"query": "hello", "k": 1})
    assert r1.status_code == 200
    assert call_count[0] == 1

    # Second identical request should use cache
    r2 = client.get("/search", params={"query": "hello", "k": 1})
    assert r2.status_code == 200
    assert call_count[0] == 1  # Should not increase

    # Different query should trigger new call
    r3 = client.get("/search", params={"query": "world", "k": 1})
    assert r3.status_code == 200
    assert call_count[0] == 2
