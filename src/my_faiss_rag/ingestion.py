"""Load the knowledge-base PDF and split it into chunks."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_faiss_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_PDF_PATH


def load_pdf(path: Path = DEFAULT_PDF_PATH) -> list[Document]:
	"""Load a PDF as one document per page."""
	return PyPDFLoader(str(path)).load()


def split_documents(
	docs: list[Document],
	chunk_size: int = CHUNK_SIZE,
	chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
	"""Split documents into overlapping chunks."""
	splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

	return splitter.split_documents(docs)
