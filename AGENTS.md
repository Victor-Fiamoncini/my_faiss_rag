# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

An HR internal-policy assistant built as a RAG (retrieval-augmented generation) study project.
Users ask questions in a Streamlit chat UI, and an OpenAI model answers them in Brazilian Portuguese,
using only the context retrieved from the company's HR policy PDF.

**Stack:** Python 3.12 · uv · Streamlit (UI) · LangChain with LCEL (orchestration) · FAISS (vector store) · OpenAI (embeddings + LLM)
**Tooling:** Ruff (format + lint) · mypy (strict) · pytest + pytest-cov · pre-commit

## Project structure

```
.
├── src/my_faiss_rag/
│   ├── app.py              # Streamlit UI (entry point)
│   ├── chain.py            # FAISS vector store + LCEL RAG chain
│   ├── ingestion.py        # PDF loading and chunking
│   └── config.py           # Paths, model names, chunk sizes, .env loading
├── tests/                  # pytest suite (offline, fakes instead of OpenAI)
├── data/document.pdf       # Knowledge-base PDF (git-ignored, bring your own)
├── .agents/skills/         # Agent skills (symlinks into the installed streamlit package)
├── .claude/skills/         # Same skills, exposed to Claude Code
├── .env.example            # Environment variables template
├── CLAUDE.md               # Claude Code entry point, imports this file
├── .env.local              # Local secrets (git-ignored) — never read or commit it
├── .pre-commit-config.yaml # Git hooks: whitespace checks, ruff, mypy
├── mise.toml               # Pinned Python and uv versions
├── pyproject.toml          # Dependencies and all tool configs
└── uv.lock                 # Locked dependency versions (commit it)
```

## Setup and commands

```bash
mise install                  # Python 3.12 + uv
uv sync                       # create .venv and install app + dev deps
uv run pre-commit install     # enable git hooks
cp .env.example .env.local    # then set OPENAI_API_KEY

uv run streamlit run src/my_faiss_rag/app.py   # http://localhost:8501

uv run ruff format .          # format
uv run ruff check --fix .     # lint
uv run mypy                   # type check (src + tests)
uv run pytest                 # tests + coverage (fails under 90%)
uv run pre-commit run --all-files
```

Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), never by editing `uv.lock`.

## How the app works

1. **Config (`config.py`):** `PROJECT_ROOT` is resolved from the package location. `.env.local` and
   `data/document.pdf` hang off it, so the app runs from any working directory. Keep new paths there.
2. **Ingestion (`ingestion.py`):** `load_pdf()` uses `PyPDFLoader`. `split_documents()` makes
   1000-character chunks with 200 overlap.
3. **Chain (`chain.py`):** `build_chain()` = load → split → `create_vector_store()` (FAISS +
   `OpenAIEmbeddings`) → `create_rag_chain()`:
   `RunnableParallel(context=retriever | format_docs, question) | prompt | ChatOpenAI("gpt-4o-mini") | StrOutputParser()`.
   `embeddings` and `llm` are injectable so tests can pass fakes.
4. **UI (`app.py`):** a plain Streamlit script. `get_chain()` is wrapped in `st.cache_resource`, so
   the PDF is embedded once per process. Chat history lives in `st.session_state["messages"]`.
   `st.chat_input` has `key="chat_input"`.

## Testing

- Tests never call OpenAI. Use the fixtures in `tests/conftest.py`: `fake_embeddings`
  (`DeterministicFakeEmbedding`), `fake_llm` (`FakeListChatModel`), `sample_pdf` (a tiny PDF
  generated on the fly, because `data/` is git-ignored).
- UI tests use `streamlit.testing.v1.AppTest` and monkeypatch `chain.build_chain`.
- Keep coverage at or above 90%.

## Conventions

- **Indentation:** tabs in Python (see `.editorconfig`; Ruff formats with `indent-style = "tab"`).
  YAML/TOML/Markdown use 2 spaces.
- **Types:** everything is type-annotated, and mypy runs in strict mode.
- **Language:** code and identifiers are in English. The prompt and user-facing UI text are in Brazilian Portuguese.
- **LCEL chains:** start chains with `RunnableParallel(...)` rather than a bare dict, so the type checker
  doesn't flag `dict | Runnable`. Values inside it must be runnables or callables, never plain strings.
- **Streamlit:** load the `developing-with-streamlit` skill before changing the UI. Prefer native
  Streamlit features. Only add CSS when explicitly asked, and target elements through `key=` classes.
- **Secrets:** the only secret is `OPENAI_API_KEY`, loaded from `.env.local`. Never hard-code or print it.

## Known issues

- `langchain-community` emits a `DeprecationWarning` on import (the package is being sunset). It's
  harmless for now (`PyPDFLoader` and `FAISS` still come from it) and is filtered in pytest.
- The skill folders are symlinks into `.venv`, so they only resolve after `uv sync`.
