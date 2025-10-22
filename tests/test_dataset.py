import lzma
from pathlib import Path

import pytest

from stfo_colbert import DELIMITER
from stfo_colbert.dataset import prepare_from_delimited_txt, prepare_from_directory
from stfo_colbert.utils import write_text_file


def test_prepare_from_delimited_txt(tmp_path: Path):
    content = f"Doc A{DELIMITER}Doc B{DELIMITER}Doc C"
    p = tmp_path / "data.txt"
    write_text_file(p, content)

    prepared = prepare_from_delimited_txt(p)
    assert prepared.document_ids == ["0", "1", "2"]
    assert prepared.documents == ["Doc A", "Doc B", "Doc C"]
    assert all(DELIMITER not in d for d in prepared.documents)


def test_prepare_from_directory_creates_cache(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.md"
    # Include a delimiter in one of the files to ensure cleaning
    write_text_file(a, f"Hello A with delimiter {DELIMITER} inside")
    write_text_file(b, "Hello B")

    prepared = prepare_from_directory(tmp_path)
    assert len(prepared.documents) == 2
    assert prepared.document_ids == ["0", "1"]
    # No internal delimiters should remain inside individual docs
    assert all(DELIMITER not in d for d in prepared.documents)

    # Verify cache file was created
    cache_path = tmp_path / ".stfo_colbert_cache.txt.xz"
    assert cache_path.exists()


def test_prepare_from_directory_reuses_cache(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.md"
    write_text_file(a, "Content A")
    write_text_file(b, "Content B")

    # First call creates cache
    prepared1 = prepare_from_directory(tmp_path)
    assert len(prepared1.documents) == 2
    cache_path = tmp_path / ".stfo_colbert_cache.txt.xz"
    assert cache_path.exists()

    # Modify files after cache creation
    write_text_file(a, "Modified A")

    # Second call should use cache (not see modifications)
    prepared2 = prepare_from_directory(tmp_path)
    assert len(prepared2.documents) == 2
    assert prepared2.documents == prepared1.documents
    assert "Modified A" not in prepared2.documents[0]


def test_cache_file_format(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    write_text_file(a, "First document")
    write_text_file(b, "Second document")

    prepared = prepare_from_directory(tmp_path)
    cache_path = tmp_path / ".stfo_colbert_cache.txt.xz"

    # Read and verify cache file contents
    with lzma.open(cache_path, 'rt', encoding='utf-8') as f:
        cache_content = f.read()

    # Cache should contain documents joined by delimiter
    assert cache_content.count(DELIMITER) == 1
    parts = cache_content.split(DELIMITER)
    assert len(parts) == 2
    assert "First document" in parts[0]
    assert "Second document" in parts[1]


def test_cleanup_does_nothing(tmp_path: Path):
    a = tmp_path / "a.txt"
    write_text_file(a, "Content")

    prepared = prepare_from_directory(tmp_path)
    cache_path = tmp_path / ".stfo_colbert_cache.txt.xz"
    assert cache_path.exists()

    # Cleanup should not delete cache file
    prepared.cleanup()
    assert cache_path.exists()
