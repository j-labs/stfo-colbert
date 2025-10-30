import os
from pathlib import Path

# Mitigate OpenMP duplicate symbol issue on macOS when using PyTorch-linked libs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def stream_text_file(path: Path, chunk_size: int = 8192):
    """Stream text file in chunks to avoid loading entire file into memory.

    Args:
        path: Path to the text file
        chunk_size: Size of chunks to read in bytes (default 8KB)

    Yields:
        str: Text chunks from the file
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def write_text_file(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)
