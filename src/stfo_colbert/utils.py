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


def write_text_file(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)
