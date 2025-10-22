import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import DELIMITER
from .utils import read_text_file, write_text_file


@dataclass(frozen=True)
class PreparedDataset:
    # Original input path (file or directory)
    source: Path
    # Temporary delimited text file created only if source was a directory.
    # None if source itself is a delimited text file.
    temp_delimited_path: Path | None
    # Document IDs and their content (kept in-memory for serving text in responses)
    document_ids: list[str]
    documents: list[str]

    def cleanup(self) -> None:
        # Remove temporary file if created
        if self.temp_delimited_path and self.temp_delimited_path.exists():
            try:
                self.temp_delimited_path.unlink()
            except Exception:
                pass


# -----------------
# Reading utilities
# -----------------


def _clean_delimiter(text: str) -> str:
    # Replace the delimiter if it appears inside a document to avoid splitting issues
    return text.replace(DELIMITER, "\n\n++++++++\n\n")


def _read_txt_or_md(path: Path) -> str:
    return _clean_delimiter(read_text_file(path))


def _read_pdf_if_available(path: Path) -> str:
    try:
        from pypdf import PdfReader  # optional
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        out = io.StringIO()
        for page in reader.pages:
            out.write(page.extract_text() or "")
            out.write("\n")
        return _clean_delimiter(out.getvalue())
    except Exception:
        return ""


def _read_docx_if_available(path: Path) -> str:
    try:
        import docx  # type: ignore  # optional python-docx
    except Exception:
        return ""
    try:
        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        return _clean_delimiter("\n".join(parts))
    except Exception:
        return ""


def _read_odt_if_available(path: Path) -> str:
    try:
        from odf.opendocument import load  # optional
        from odf.text import P
    except Exception:
        return ""
    try:
        txts: list[str] = []
        doc = load(str(path))
        for p in doc.getElementsByType(P):
            if p.firstChild is not None:
                txts.append(str(p.firstChild.data))
        return _clean_delimiter("\n".join(txts))
    except Exception:
        return ""


# -----------------
# Public API
# -----------------


def prepare_from_delimited_txt(path: Path) -> PreparedDataset:
    raw = read_text_file(path)
    parts = [p.strip() for p in raw.split(DELIMITER)]
    docs = [p for p in parts if p]
    ids = [str(i) for i in range(len(docs))]
    return PreparedDataset(source=path, temp_delimited_path=None, document_ids=ids, documents=docs)


def prepare_from_directory(dir_path: Path) -> PreparedDataset:
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    docs: list[str] = []
    for p in files:
        suffix = p.suffix.lower()
        text = ""
        if suffix in {".txt", ".md"}:
            text = _read_txt_or_md(p)
        elif suffix == ".pdf":
            text = _read_pdf_if_available(p)
        elif suffix == ".docx":
            text = _read_docx_if_available(p)
        elif suffix == ".odt":
            text = _read_odt_if_available(p)
        # Skip empty extractions
        if text.strip():
            docs.append(text.strip())

    ids = [str(i) for i in range(len(docs))]

    # Write a temporary delimited file for transparency/traceability (deleted later)
    tmp = tempfile.NamedTemporaryFile(prefix="stfo_colbert_", suffix=".txt", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    write_text_file(tmp_path, DELIMITER.join(docs))

    return PreparedDataset(source=dir_path, temp_delimited_path=tmp_path, document_ids=ids, documents=docs)


def prepare_dataset(path: Path) -> PreparedDataset:
    path = path.expanduser().resolve()
    if path.is_dir():
        return prepare_from_directory(path)
    if path.is_file():
        return prepare_from_delimited_txt(path)
    raise FileNotFoundError(f"Dataset path does not exist: {path}")
