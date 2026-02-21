# DSPy Classification Studio

Production-ready text analysis platform with:

- Flask MVC backend (`views`/`controllers`/`models`/`services`)
- AngularJS MVC frontend
- LangGraph multi-step agent
- Knowledge graph construction + inference
- Deterministic rule-based fallback (works without remote LLM)

## Architecture

```text
Browser (AngularJS MVC)
  -> Flask Routes (app/views)
  -> Controller Orchestration (app/controllers)
  -> Analysis Engines + Graph Services (app/services)
  -> Schemas + Classifier Signatures (app/models)
  -> LangGraph Agent (app/agents)
```

## Core capabilities

- Single and batch classification (`sentiment`, `topic`, `intent`, `multi_label`, `entity`)
- Agent workflow with graph enrichment:
  - sentiment -> topic -> intent -> entities -> knowledge graph -> summary
- Entity-centric graph inference:
  - neighbor traversal (`max_depth`)
  - optional relation filtering
  - predicted links via two-hop reasoning
- Persistent graph storage to `data/graph/knowledge_graph.json`

## Project layout

```text
.
├── app/
│   ├── agents/          # LangGraph pipeline
│   ├── controllers/     # Application orchestration
│   ├── domain/          # Enums and domain errors
│   ├── models/          # Pydantic schemas + DSPy signatures
│   ├── services/        # DSPy service, text engines, knowledge graph
│   ├── static/          # AngularJS JS + CSS
│   ├── templates/       # HTML views
│   └── views/           # Flask routes
├── tests/               # Unit/integration tests
├── .github/workflows/   # CI/CD pipelines
├── Dockerfile
├── docker-compose.yml
├── config.py
├── run.py
└── wsgi.py
```

## Local run

```bash
python -m pip install -r requirements-dev.txt
cp .env.example .env
python run.py --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

## Docker

```bash
docker compose up --build
```

Provider selection in Docker compose:

- default: `rule_based` (no external model required)
- use Ollama: `APP_PROVIDER=ollama docker compose up --build`
- custom Ollama URL: `APP_OLLAMA_BASE_URL=http://host.docker.internal:11434`

Run tests in Docker:

```bash
docker compose run --rm test
```

## API endpoints

- `GET /health`
- `GET /api/classifiers`
- `POST /api/classify`
- `POST /api/classify/batch`
- `POST /api/agent/analyze`
- `GET /api/knowledge-graph`
- `POST /api/graph/infer`

### Graph inference request example

```json
{
  "entity": "Python",
  "entity_type": "CONCEPT",
  "max_depth": 3,
  "relation_filter": "co_occurs"
}
```

## Quality and CI/CD

CI validates:

- lint (`flake8`)
- tests with coverage threshold (`>=80%`)
- Docker production image build
- Docker smoke test against `/health`

CD publishes multi-arch images to GHCR on `v*` tags.

## Useful commands

```bash
make lint
make test
make test-cov
make ci
make docker-build
make docker-up
make docker-test
```
