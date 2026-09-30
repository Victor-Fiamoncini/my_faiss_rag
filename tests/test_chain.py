from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from my_faiss_rag import chain
from my_faiss_rag.config import CHAT_MODEL, EMBEDDING_MODEL
from tests.conftest import SAMPLE_TEXT

CHUNKS = [
	Document(page_content="Férias: 30 dias por ano."),
	Document(page_content="Home office: até 3 dias por semana."),
]


def echo_prompt(prompt: PromptValue) -> AIMessage:
	"""Stand-in LLM that answers with the prompt it received."""
	return AIMessage(content=prompt.to_string())


def test_create_vector_store_retrieves_indexed_chunk(fake_embeddings: Embeddings) -> None:
	store = chain.create_vector_store(CHUNKS, fake_embeddings)

	# DeterministicFakeEmbedding maps identical text to identical vectors.
	results = store.similarity_search(CHUNKS[1].page_content, k=1)

	assert results[0].page_content == CHUNKS[1].page_content


def test_create_vector_store_defaults_to_openai_embeddings(
	monkeypatch: pytest.MonkeyPatch, fake_embeddings: Embeddings
) -> None:
	models: list[str] = []

	def fake_openai_embeddings(model: str) -> Embeddings:
		models.append(model)
		return fake_embeddings

	monkeypatch.setattr(chain, "OpenAIEmbeddings", fake_openai_embeddings)

	chain.create_vector_store(CHUNKS)

	assert models == [EMBEDDING_MODEL]


def test_format_docs_joins_page_content() -> None:
	assert chain.format_docs(CHUNKS) == (
		"Férias: 30 dias por ano.\n\nHome office: até 3 dias por semana."
	)


def test_format_docs_empty() -> None:
	assert chain.format_docs([]) == ""


def test_rag_chain_returns_llm_answer(
	fake_embeddings: Embeddings, fake_llm: FakeListChatModel
) -> None:
	retriever = chain.create_vector_store(CHUNKS, fake_embeddings).as_retriever()

	rag_chain = chain.create_rag_chain(retriever, fake_llm)

	assert rag_chain.invoke("Quantos dias de férias?") == "Resposta de teste."


def test_rag_chain_prompt_has_question_and_context(fake_embeddings: Embeddings) -> None:
	retriever = chain.create_vector_store(CHUNKS, fake_embeddings).as_retriever(
		search_kwargs={"k": 2}
	)

	rag_chain = chain.create_rag_chain(retriever, RunnableLambda(echo_prompt))  # type: ignore[arg-type]
	prompt = rag_chain.invoke("Quantos dias de férias?")

	assert "Pergunta: Quantos dias de férias?" in prompt
	assert "Contexto: " in prompt
	for doc in CHUNKS:
		assert doc.page_content in prompt
	assert "page_content=" not in prompt


def test_rag_chain_defaults_to_openai_chat_model(fake_embeddings: Embeddings) -> None:
	retriever = chain.create_vector_store(CHUNKS, fake_embeddings).as_retriever()

	rag_chain = chain.create_rag_chain(retriever)

	llms = [step for step in rag_chain.steps if isinstance(step, BaseChatModel)]  # type: ignore[attr-defined]
	assert len(llms) == 1
	assert isinstance(llms[0], ChatOpenAI)
	assert llms[0].model_name == CHAT_MODEL


def test_build_chain_end_to_end(
	sample_pdf: Path, fake_embeddings: Embeddings, fake_llm: FakeListChatModel
) -> None:
	rag_chain = chain.build_chain(sample_pdf, fake_embeddings, fake_llm)

	assert rag_chain.invoke("Quantos dias de férias?") == "Resposta de teste."


def test_build_chain_uses_pdf_as_context(sample_pdf: Path, fake_embeddings: Embeddings) -> None:
	rag_chain = chain.build_chain(sample_pdf, fake_embeddings, RunnableLambda(echo_prompt))  # type: ignore[arg-type]

	assert SAMPLE_TEXT in rag_chain.invoke("Quantos dias de férias?")
