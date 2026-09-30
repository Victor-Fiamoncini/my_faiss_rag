"""Project paths, model names and environment loading."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env.local"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PDF_PATH = DATA_DIR / "document.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def load_env(path: Path = ENV_FILE) -> bool:
	"""Load environment variables from `path`. Returns whether any were set."""
	return load_dotenv(path)
