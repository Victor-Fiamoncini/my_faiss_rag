from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.runnables import Runnable, RunnableLambda
from streamlit.testing.v1 import AppTest

from my_faiss_rag import chain

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "my_faiss_rag" / "app.py"


def fake_build_chain() -> Runnable[str, str]:
	return RunnableLambda(lambda question: f"Resposta para: {question}")


@pytest.fixture(autouse=True)
def patched_chain(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
	monkeypatch.setattr(chain, "build_chain", fake_build_chain)
	st.cache_resource.clear()
	yield
	st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
	return AppTest.from_file(str(APP_PATH), default_timeout=10).run()


def test_renders_title_and_empty_history(app: AppTest) -> None:
	assert not app.exception
	assert app.title[0].value == "Assistente RAG — Políticas Internas RH"
	assert app.session_state["messages"] == []
	assert len(app.chat_message) == 0


def test_question_gets_answered(app: AppTest) -> None:
	app.chat_input(key="chat_input").set_value("Quantos dias de férias?").run()

	assert not app.exception
	assert [m.name for m in app.chat_message] == ["user", "assistant"]
	assert app.chat_message[0].markdown[0].value == "Quantos dias de férias?"
	assert app.chat_message[1].markdown[0].value == "Resposta para: Quantos dias de férias?"


def test_history_persists_across_questions(app: AppTest) -> None:
	app.chat_input(key="chat_input").set_value("Primeira").run()
	app.chat_input(key="chat_input").set_value("Segunda").run()

	assert not app.exception
	assert app.session_state["messages"] == [
		{"role": "user", "content": "Primeira"},
		{"role": "assistant", "content": "Resposta para: Primeira"},
		{"role": "user", "content": "Segunda"},
		{"role": "assistant", "content": "Resposta para: Segunda"},
	]
	assert len(app.chat_message) == 4
