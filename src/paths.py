from pathlib import Path

def find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:   # [current, current.parents[0], current.parents[1], current.parents[2], ...]
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(...)

PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"