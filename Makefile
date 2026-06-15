.PHONY: install install-dev test lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest tests/ -v

lint:
	ruff check src/ examples/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
