# Contributing to DSPy Classification Studio

## Development Setup

```bash
git clone <repo-url>
cd PythonProject8
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

## Running Locally

```bash
# Rule-based (no external dependencies)
PROVIDER=rule_based python run.py

# With Ollama
ollama pull llama3.2:3b
python run.py
```

## Code Quality

Before submitting a PR, run the full CI suite:

```bash
make ci
```

This runs:
1. **Ruff** — fast Python linter
2. **Flake8** — style enforcement
3. **Tests** — pytest with 80%+ coverage threshold

## Testing

```bash
make test          # Quick run
make test-cov      # With coverage report
```

All tests run against the `rule_based` provider so no external LLM is needed.

## Architecture

The project follows **MVC + DDD** patterns:

- **Models** (`app/models/`) — Pydantic schemas, DSPy signatures, BootstrapFewShot optimizer
- **Views** (`app/views/`) — Flask routes
- **Controllers** (`app/controllers/`) — Business logic orchestration
- **Services** (`app/services/`) — Engine implementations, knowledge graph
- **Agents** (`app/agents/`) — LangGraph pipelines
- **Domain** (`app/domain/`) — Enums, errors, value objects

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `test:` — Test changes
- `refactor:` — Code restructuring
- `ci:` — CI/CD changes

## Releasing

Tag a version to trigger CD:

```bash
git tag v2.1.0
git push origin v2.1.0
```

This builds and pushes multi-arch Docker images to GHCR.

