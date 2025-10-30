from pathlib import Path
import pytest

from stfo_colbert.indexer import (
    CollectionDB,
    load_collection,
)


def test_collection_db_basic_operations(tmp_path: Path):
    """Test basic CollectionDB operations: add, get, len, iter."""
    db_path = tmp_path / "test.db"

    with CollectionDB(db_path) as db:
        # Test adding documents
        db.add("0", "First document")
        db.add("1", "Second document")
        db.add("2", "Third document")
        db.commit()

        # Test __getitem__
        assert db["0"] == "First document"
        assert db["1"] == "Second document"
        assert db["2"] == "Third document"

        # Test get method
        assert db.get("0") == "First document"
        assert db.get("999") is None
        assert db.get("999", "default") == "default"

        # Test __len__
        assert len(db) == 3

        # Test __iter__
        keys = sorted(list(db))
        assert keys == ["0", "1", "2"]

        # Test KeyError for missing key
        with pytest.raises(KeyError):
            _ = db["999"]


def test_collection_db_batch_operations(tmp_path: Path):
    """Test batch insert operations for efficiency."""
    db_path = tmp_path / "test_batch.db"

    with CollectionDB(db_path) as db:
        # Add 1000 documents in batch
        batch = [(str(i), f"Document {i}") for i in range(1000)]
        db.add_batch(batch)
        db.commit()

        # Verify count
        assert len(db) == 1000

        # Verify some random entries
        assert db["0"] == "Document 0"
        assert db["500"] == "Document 500"
        assert db["999"] == "Document 999"


def test_collection_db_update_existing(tmp_path: Path):
    """Test updating existing documents (INSERT OR REPLACE)."""
    db_path = tmp_path / "test_update.db"

    with CollectionDB(db_path) as db:
        db.add("0", "Original text")
        db.commit()

        # Update the same key
        db.add("0", "Updated text")
        db.commit()

        # Should have only one entry with updated text
        assert len(db) == 1
        assert db["0"] == "Updated text"


def test_collection_db_persistence(tmp_path: Path):
    """Test that data persists after closing and reopening."""
    db_path = tmp_path / "test_persist.db"

    # Write data
    with CollectionDB(db_path) as db:
        db.add("0", "Persistent document")
        db.add("1", "Another document")
        db.commit()

    # Reopen and verify
    with CollectionDB(db_path) as db:
        assert len(db) == 2
        assert db["0"] == "Persistent document"
        assert db["1"] == "Another document"


def test_load_collection_missing_file(tmp_path: Path):
    """Test load_collection returns None when file doesn't exist."""
    result = load_collection(tmp_path)
    assert result is None


def test_collection_db_appending_to_existing(tmp_path: Path):
    """Test appending new documents to an existing database."""
    db_path = tmp_path / "test_append.db"

    # Create initial database
    with CollectionDB(db_path) as db:
        db.add("0", "First document")
        db.add("1", "Second document")
        db.commit()

    # Reopen and append more documents
    with CollectionDB(db_path) as db:
        assert len(db) == 2

        # Append new documents
        db.add("2", "Third document")
        db.add("3", "Fourth document")
        db.commit()

        assert len(db) == 4

    # Verify all documents are present
    with CollectionDB(db_path) as db:
        assert len(db) == 4
        assert db["0"] == "First document"
        assert db["1"] == "Second document"
        assert db["2"] == "Third document"
        assert db["3"] == "Fourth document"


def test_collection_db_large_batch_streaming(tmp_path: Path):
    """Test streaming large batches (simulating real-world usage)."""
    db_path = tmp_path / "test_large.db"

    # Simulate streaming 50k documents
    def document_generator():
        for i in range(50000):
            yield f"Document number {i} with some text content"

    with CollectionDB(db_path) as db:
        batch = []
        batch_size = 10000

        for idx, doc in enumerate(document_generator()):
            batch.append((str(idx), doc))

            if len(batch) >= batch_size:
                db.add_batch(batch)
                db.commit()
                batch.clear()

        # Add remaining
        if batch:
            db.add_batch(batch)
            db.commit()

    # Verify
    with CollectionDB(db_path) as db:
        assert len(db) == 50000
        assert db["0"] == "Document number 0 with some text content"
        assert db["25000"] == "Document number 25000 with some text content"
        assert db["49999"] == "Document number 49999 with some text content"


def test_collection_db_as_mapping(tmp_path: Path):
    """Test that CollectionDB works as a Mapping for type compatibility."""
    from collections.abc import Mapping

    db_path = tmp_path / "test_mapping.db"

    with CollectionDB(db_path) as db:
        db.add("key1", "value1")
        db.commit()

        # Should be instance of Mapping
        assert isinstance(db, Mapping)

        # Should work with dict-like operations
        assert "key1" in db
        assert "missing" not in db
