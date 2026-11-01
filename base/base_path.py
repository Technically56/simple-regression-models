from pathlib import Path

def get_base_path() -> Path:
    """Return the absolute path to the base directory of the project."""
    return Path(__file__).resolve().parent.parent