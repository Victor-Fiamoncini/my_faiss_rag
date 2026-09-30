from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

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


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Keep tests offline: a dummy key so OpenAI clients can be built but never used."""
	monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


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
