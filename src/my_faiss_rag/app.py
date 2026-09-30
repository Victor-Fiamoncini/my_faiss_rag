"""Streamlit chat UI for the HR policy assistant."""

import streamlit as st
from langchain_core.runnables import Runnable

from my_faiss_rag import chain
from my_faiss_rag.config import load_env

load_env()


@st.cache_resource(show_spinner="Indexando o documento...")
def get_chain() -> Runnable[str, str]:
	return chain.build_chain()


st.title("Assistente RAG — Políticas Internas RH")

if "messages" not in st.session_state:
	st.session_state["messages"] = []

for msg in st.session_state["messages"]:
	with st.chat_message(msg["role"]):
		st.write(msg["content"])

prompt = st.chat_input("Pergunte sobre as políticas da empresa...", key="chat_input")

if prompt:
	st.session_state["messages"].append({"role": "user", "content": prompt})

	with st.chat_message("user"):
		st.write(prompt)

	with st.chat_message("assistant"), st.spinner("Buscando..."):
		answer = get_chain().invoke(prompt)

		st.write(answer)

	st.session_state["messages"].append({"role": "assistant", "content": answer})
