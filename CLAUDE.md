# CLAUDE.md

Project guidance lives in `AGENTS.md` (shared by all coding agents) and is imported here:

@AGENTS.md

## Claude Code notes

- Run every Python tool through uv (`uv run pytest`, `uv run mypy`, ...). If `uv` isn't on `PATH`, prefix with `mise exec --`.
- Before finishing a change, run `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`, and report failures instead of hiding them.
- Load the `developing-with-streamlit` skill (in `.claude/skills/`) before touching `src/my_faiss_rag/app.py`.
- Never read `.env.local` or make real OpenAI calls in tests. Use the fakes in `tests/conftest.py`.
- Keep `AGENTS.md` as the single source of truth: put shared guidance there, and only Claude-specific notes here.
