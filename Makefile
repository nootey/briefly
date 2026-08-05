default: run

run:
	uv run python -m main

lint:
	uv run --group dev ruff check . --fix
