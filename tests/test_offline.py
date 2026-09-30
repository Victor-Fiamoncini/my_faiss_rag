"""Guards that keep the test suite from reaching OpenAI or reading real secrets."""

import os
import socket

import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pytest_socket import SocketBlockedError

from my_faiss_rag import config
from tests.conftest import OFFLINE_BASE_URL, UNSAFE_ENV_VARS


def test_network_is_blocked() -> None:
	with pytest.raises(SocketBlockedError):
		socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_openai_clients_point_nowhere() -> None:
	assert os.environ["OPENAI_API_KEY"] == "sk-test"
	# Built without explicit settings, like the app's defaults, so they read the environment.
	assert str(ChatOpenAI().root_client.base_url) == f"{OFFLINE_BASE_URL}/"
	assert str(OpenAIEmbeddings().client._client.base_url) == f"{OFFLINE_BASE_URL}/"


def test_tracing_and_account_vars_are_cleared() -> None:
	assert os.environ["LANGSMITH_TRACING"] == "false"
	assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
	assert not any(name in os.environ for name in UNSAFE_ENV_VARS)


def test_default_load_env_skips_real_env_file() -> None:
	assert config.load_env() is False
