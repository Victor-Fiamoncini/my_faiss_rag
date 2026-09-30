from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from my_faiss_rag import config

SAMPLE_TEXT = "Os colaboradores tem direito a 30 dias de ferias por ano."


def build_pdf(text: str) -> bytes:
	"""Build a minimal single-page PDF showing `text` in Helvetica."""
	stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
	objects = [
		b"<< /Type /Catalog /Pages 2 0 R >>",
		b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
		b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
		b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
		b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
	]

	pdf = b"%PDF-1.4\n"
	offsets = []
	for number, body in enumerate(objects, start=1):
		offsets.append(len(pdf))
		pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)

	xref_offset = len(pdf)
	pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
	pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
	pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
		len(objects) + 1,
		xref_offset,
	)

	return pdf


# Anything that could point a client at a real account or a tracing backend.
UNSAFE_ENV_VARS = (
	"OPENAI_ORG_ID",
	"OPENAI_ORGANIZATION",
	"OPENAI_PROJECT_ID",
	"LANGSMITH_API_KEY",
	"LANGSMITH_ENDPOINT",
	"LANGCHAIN_API_KEY",
	"LANGCHAIN_ENDPOINT",
)
OFFLINE_BASE_URL = "http://openai.invalid/v1"  # .invalid never resolves (RFC 2606)


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	"""Keep tests away from OpenAI, LangSmith and the developer's `.env.local`.

	Network access is blocked by pytest-socket (see `addopts`); this is the second layer.
	"""
	for name in UNSAFE_ENV_VARS:
		monkeypatch.delenv(name, raising=False)
	monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
	monkeypatch.setenv("OPENAI_BASE_URL", OFFLINE_BASE_URL)
	monkeypatch.setenv("LANGSMITH_TRACING", "false")
	monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

	# `app.py` calls `load_env()`: point its default at a missing file instead of the real one.
	real_load_env = config.load_env
	missing_env_file = tmp_path / ".env.local"

	def isolated_load_env(path: Path = missing_env_file) -> bool:
		return real_load_env(path)

	monkeypatch.setattr(config, "load_env", isolated_load_env)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
	path = tmp_path / "sample.pdf"
	path.write_bytes(build_pdf(SAMPLE_TEXT))

	return path


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
	return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def fake_llm() -> FakeListChatModel:
	return FakeListChatModel(responses=["Resposta de teste."])
