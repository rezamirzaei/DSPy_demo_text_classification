.PHONY: install install-dev lint ruff type-check test test-cov run ci docker-build docker-up docker-test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements-runtime.txt

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt

lint:
	flake8 config.py app/ tests/ --max-line-length 120 --ignore E501,W503

ruff:
	ruff check config.py app/ tests/

type-check:
	mypy config.py app/ --ignore-missing-imports --no-error-summary || true

test:
	PROVIDER=rule_based ENVIRONMENT=test python -m pytest tests/ -v --tb=short

test-cov:
	PROVIDER=rule_based ENVIRONMENT=test python -m pytest tests/ -v --tb=short --cov=app --cov=config --cov-report=term-missing --cov-report=xml --cov-fail-under=80

run:
	python run.py --host 0.0.0.0 --port 8000

ci: ruff lint test-cov

docker-build:
	DOCKER_BUILDKIT=1 docker build --target production -t dspy-studio:latest .

docker-up:
	DOCKER_BUILDKIT=1 docker compose up --build

docker-test:
	DOCKER_BUILDKIT=1 docker compose run --rm test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov coverage.xml .mypy_cache
