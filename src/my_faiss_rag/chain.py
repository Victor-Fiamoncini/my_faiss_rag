"""Vector store and LCEL RAG chain."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from my_faiss_rag.config import CHAT_MODEL, DEFAULT_PDF_PATH, EMBEDDING_MODEL
from my_faiss_rag.ingestion import load_pdf, split_documents

PROMPT_TEMPLATE = """
	Você é um assistente de RH que responde perguntas sobre políticas internas da empresa.
	Use APENAS as informações do contexto abaixo para responder.
	Se não encontrar a resposta, diga claramente que não sabe responder.
	Responda em português do Brasil, de forma clara e objetiva.

	Contexto: {context}
	Pergunta: {question}
	Resposta:
"""


def create_vector_store(chunks: list[Document], embeddings: Embeddings | None = None) -> FAISS:
	"""Index the chunks in an in-memory FAISS store."""
	return FAISS.from_documents(chunks, embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL))


def format_docs(docs: list[Document]) -> str:
	"""Join the retrieved documents' text into a single context string."""
	return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(
	retriever: BaseRetriever, llm: BaseChatModel | None = None
) -> Runnable[str, str]:
	"""Build the chain: question -> retrieve context -> prompt -> LLM -> text."""
	prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

	return (
		RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
		| prompt
		| (llm or ChatOpenAI(model=CHAT_MODEL))
		| StrOutputParser()
	)


def build_chain(
	pdf_path: Path = DEFAULT_PDF_PATH,
	embeddings: Embeddings | None = None,
	llm: BaseChatModel | None = None,
) -> Runnable[str, str]:
	"""Load, split and index the PDF, then build the RAG chain over it."""
	chunks = split_documents(load_pdf(pdf_path))

	return create_rag_chain(create_vector_store(chunks, embeddings).as_retriever(), llm)
