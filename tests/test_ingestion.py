from pathlib import Path

from langchain_core.documents import Document

from my_faiss_rag.ingestion import load_pdf, split_documents
from tests.conftest import SAMPLE_TEXT


def test_load_pdf_returns_one_document_per_page(sample_pdf: Path) -> None:
	docs = load_pdf(sample_pdf)

	assert len(docs) == 1
	assert SAMPLE_TEXT in docs[0].page_content
	assert docs[0].metadata["page"] == 0
	assert docs[0].metadata["source"] == str(sample_pdf)


def test_split_documents_respects_chunk_size_and_overlap() -> None:
	text = " ".join(f"palavra{i:03d}" for i in range(200))

	chunks = split_documents([Document(page_content=text)], chunk_size=100, chunk_overlap=20)

	assert len(chunks) > 1
	assert all(len(chunk.page_content) <= 100 for chunk in chunks)
	first_tail = chunks[0].page_content.split()[-1]
	assert chunks[1].page_content.startswith(first_tail)


def test_split_documents_keeps_short_document_whole() -> None:
	doc = Document(page_content="curto", metadata={"page": 3})

	chunks = split_documents([doc])

	assert chunks == [doc]


def test_split_documents_empty_list() -> None:
	assert split_documents([]) == []
