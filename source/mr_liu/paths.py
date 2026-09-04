from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of ``source/``)."""
    return Path(__file__).resolve().parents[2]


def ensure_source_on_path() -> Path:
    """Insert ``source/`` on ``sys.path`` and return it."""
    import sys

    source = repo_root() / "source"
    source_str = str(source)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    return source
