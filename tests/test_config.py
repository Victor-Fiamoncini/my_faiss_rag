import os
from pathlib import Path

import pytest

from my_faiss_rag import config


def test_paths_point_to_project_root() -> None:
	assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
	assert config.ENV_FILE == config.PROJECT_ROOT / ".env.local"
	assert config.DEFAULT_PDF_PATH == config.PROJECT_ROOT / "data" / "document.pdf"


def test_load_env_reads_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("MY_FAISS_RAG_TEST_VAR", raising=False)
	env_file = tmp_path / ".env"
	env_file.write_text("MY_FAISS_RAG_TEST_VAR=hello\n")

	assert config.load_env(env_file) is True
	assert os.environ["MY_FAISS_RAG_TEST_VAR"] == "hello"

	monkeypatch.delenv("MY_FAISS_RAG_TEST_VAR")


def test_load_env_missing_file(tmp_path: Path) -> None:
	assert config.load_env(tmp_path / "missing.env") is False
