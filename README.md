# DSPy Classification Studio

Production-ready text analysis platform powered by DSPy and LangGraph.

## What it does

Analyses any text through a multi-step LangGraph agent that classifies
sentiment, topic and intent, extracts entities, enriches them against a
real-world knowledge graph, and produces a quality-scored summary — all
in a single request.

## Architecture

```text
Browser (AngularJS MVC)
  → Flask Routes          (app/views)
  → Controller            (app/controllers)
  → Analysis Engines      (app/services)
  → LangGraph Agent       (app/agents)
  → Knowledge Graph       (app/services/knowledge_graph)
  → Schemas / Signatures  (app/models)
```

## Knowledge graph

The knowledge graph ships with **real-world open-source data** merged from
three sources:

| Source | Type | Licence |
|--------|------|---------|
| Wikidata SPARQL | Programming languages, AI models, tech companies, people | CC0 |
| Commonsense triples | 65+ hand-curated ConceptNet-style facts (is_a, used_for, …) | Project |
| Curated AI/ML graph | 110+ entities covering models, frameworks, researchers, papers | Project |

Rebuild it any time:

```bash
python scripts/build_knowledge_graph.py
```

## LangGraph agent pipeline

```text
router → sentiment → topic → intent → entities
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                         kg_enrich               (skip KG)
                              │
                          kg_build
                              └───────────┬───────────┘
                                          │
                                      summarise
                                          │
                                    quality_check
                                          │
                                         END
```

Each node is fault-tolerant: if one step fails the pipeline continues with
fallback values and the quality-check step records every issue.

## Core capabilities

- Single and batch classification (`sentiment`, `topic`, `intent`, `multi_label`, `entity`)
- Agent workflow with knowledge-graph enrichment and quality scoring
- Entity-centric graph inference with neighbour traversal, relation filtering,
  and two-hop link prediction
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
├── scripts/             # Data pipeline scripts
├── tests/               # Unit / integration tests
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
- use Ollama: `PROVIDER=ollama docker compose up --build`
- custom Ollama URL: `APP_OLLAMA_BASE_URL=http://host.docker.internal:11434`

Run tests in Docker:

```bash
docker compose run --rm test
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/api/classifiers` | List available classifiers |
| `POST` | `/api/classify` | Single-text classification |
| `POST` | `/api/classify/batch` | Batch classification |
| `POST` | `/api/agent/analyze` | Full agent analysis |
| `GET`  | `/api/knowledge-graph` | Export graph |
| `POST` | `/api/knowledge-graph/seed` | Re-seed graph with curated data |
| `POST` | `/api/graph/infer` | Entity-centric inference |

### Agent analysis request

```json
{
  "text": "DSPy is a great framework for building AI classification systems",
  "enable_knowledge_graph": true
}
```

### Graph inference request

```json
{
  "entity": "BERT",
  "entity_type": "MODEL",
  "max_depth": 3,
  "relation_filter": "used_for"
}
```

## Quality and CI/CD

CI validates:

- lint (`flake8`)
- tests with coverage threshold (≥80 %)
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
