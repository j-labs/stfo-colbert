from __future__ import annotations

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
    assert prepared.temp_delimited_path is None
    assert prepared.document_ids == ["0", "1", "2"]
    assert prepared.documents == ["Doc A", "Doc B", "Doc C"]
    assert all(DELIMITER not in d for d in prepared.documents)


def test_prepare_from_directory_creates_temp(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.md"
    # Include a delimiter in one of the files to ensure cleaning
    write_text_file(a, f"Hello A with delimiter {DELIMITER} inside")
    write_text_file(b, "Hello B")

    prepared = prepare_from_directory(tmp_path)
    try:
        assert prepared.temp_delimited_path is not None
        assert prepared.temp_delimited_path.exists()
        assert len(prepared.documents) == 2
        assert prepared.document_ids == ["0", "1"]
        # No internal delimiters should remain inside individual docs
        assert all(DELIMITER not in d for d in prepared.documents)
        # Temp file should contain the delimiter only between documents
        temp_text = prepared.temp_delimited_path.read_text(encoding="utf-8")
        assert temp_text.count(DELIMITER) == 1
    finally:
        prepared.cleanup()
        # should be deleted
        assert not prepared.temp_delimited_path.exists()
