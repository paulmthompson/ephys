# Agent guidelines

This repository is a Python project for neuroscience electrophysiology data analysis.

## Package management

Use **[uv](https://docs.astral.sh/uv/)** for all dependency work: installing the project, adding or upgrading packages, and locking versions.

- Sync the environment from the lockfile: `uv sync`
- Include development tools (pytest, Ruff, ty): `uv sync --all-groups`
- Add a runtime dependency: `uv add <package>`
- Add a dev-only dependency: `uv add --dev <package>` (then commit the updated `pyproject.toml` and `uv.lock`)

Do not rely on ad-hoc `pip install` for project development; prefer `uv run …` so commands use the project environment.

## After editing Python files

1. **Lint (Ruff):** `uv run ruff check .` (runs Ruff from the uv-managed environment).
2. **Type check (ty):** `uv run ty check .` (runs Astral’s **ty** checker the same way).

Run both after substantive changes to Python sources (and fix any new issues before considering the work complete).

`ty` is configured (via `pyproject.toml`) to analyze `data_wrangling/`, `processing/`, and `tests/` only. Standalone `scripts/` are skipped until imports are aligned with the installable package layout.

## Tests

Run the test suite with **pytest** via uv:

```bash
uv run pytest tests
```

The default `--dirpath` is `tests/data_wrangling/data` (see `tests/conftest.py`). Pass `--dirpath=…` only when pointing at other fixture directories.

Use `uv run pytest` so the same interpreter and dependencies as `uv sync` are used.
