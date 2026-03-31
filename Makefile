.PHONY: setup run test lint clean help

PYTHON    ?= python3
VENV      := venv
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
PYEXEC    := $(VENV)/bin/python

# Default target
help:
	@echo "Usage:"
	@echo "  make setup           — create venv and install dependencies"
	@echo "  make run FOLDER=...  — transcribe all media under FOLDER"
	@echo "  make test            — run the test suite"
	@echo "  make lint            — run ruff linter (if installed)"
	@echo "  make clean           — remove venv and cache files"

setup:
	bash setup.sh

run:
ifndef FOLDER
	$(error FOLDER is not set. Usage: make run FOLDER=/path/to/media)
endif
	$(PYEXEC) main.py "$(FOLDER)"

test:
	$(PYTEST) tests/ -v

lint:
	$(VENV)/bin/ruff check src/ main.py || true

clean:
	rm -rf $(VENV) __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
