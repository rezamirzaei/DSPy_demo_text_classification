# DSPy Classification Studio

Production-ready text analysis platform powered by **DSPy**, **LangGraph**, and
a real-world **Knowledge Graph**.

## What it does

Analyses any text through a multi-step LangGraph agent that runs
sentiment, topic, and intent classification **in parallel** (true
LangGraph fan-out), extracts entities, enriches them against a
curated knowledge graph of 110+ AI/ML entities, and produces a
quality-scored summary — all in a single request.

### Key differentiators

| Feature | Implementation |
|---------|---------------|
| **DSPy Optimization** | `BootstrapFewShot` with labelled training examples — automatic prompt optimization, not just structured LLM calls |
| **Parallel fan-out** | Sentiment, topic, and intent run concurrently via LangGraph's native parallel edges |
| **Knowledge Graph** | 110+ real-world entities (Wikidata + curated AI/ML) with two-hop link prediction and BFS traversal |
| **5 LLM providers** | Ollama (local), Google Gemini, OpenAI, HuggingFace, rule-based fallback |
| **Graceful degradation** | Every agent node fault-tolerant; hybrid engine with timeout-guarded primary + deterministic fallback |
| **Full CI/CD** | Lint (ruff + flake8) → type-check (mypy) → security audit → tests (80%+ coverage) → Docker build → smoke test → CD to GHCR |

## Architecture

```text
Browser (AngularJS MVC)
  → Flask Routes          (app/views)
  → Controller            (app/controllers)
  → Analysis Engines      (app/services)
  → LangGraph Agent       (app/agents)         ← parallel fan-out
  → DSPy Optimizer        (app/models)          ← BootstrapFewShot
  → Knowledge Graph       (app/services)        ← BFS + link prediction
  → Schemas / Signatures  (app/models)
```

## LangGraph agent pipeline

```text
                 router
                   │
         ┌────────┼────────┐
         ▼        ▼        ▼
     sentiment  topic   intent     ← PARALLEL
         └────────┼────────┘
             merge_analyses
                   │
              entities
                   │
         ┌────────┴────────┐
         ▼                 ▼
     kg_enrich         (skip KG)
         │
      kg_build
         └────────┬────────┘
                  │
             summarise
                  │
           quality_check
                  │
                 END
```

Each node is fault-tolerant: if one step fails the pipeline continues with
fallback values and the quality-check step records every issue.

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

## DSPy optimization

The project uses DSPy's `BootstrapFewShot` optimizer with curated training
examples for sentiment, topic, and intent classification.  Optimized
modules are cached to `data/dspy_optimized/` so the bootstrap only runs
once:

```python
from app.models.optimizer import DSPyOptimizer
optimizer = DSPyOptimizer()
optimized_modules = optimizer.optimize_all_classifiers()
```

## Core capabilities

- Single and batch classification (`sentiment`, `topic`, `intent`, `multi_label`, `entity`)
- Agent workflow with **parallel classification** and knowledge-graph enrichment
- DSPy prompt optimization via BootstrapFewShot
- Entity-centric graph inference with neighbour traversal, relation filtering,
  and two-hop link prediction
- 5 LLM provider backends with automatic fallback
- Persistent graph storage to `data/graph/knowledge_graph.json`

## Project layout

```text
.
├── app/
│   ├── agents/          # LangGraph pipeline (parallel fan-out)
│   ├── controllers/     # Application orchestration
│   ├── domain/          # Enums and domain errors
│   ├── models/          # Pydantic schemas, DSPy signatures, optimizer
│   ├── services/        # DSPy service, text engines, knowledge graph
│   ├── static/          # AngularJS JS + CSS
│   ├── templates/       # HTML views
│   └── views/           # Flask routes
├── scripts/             # Data pipeline scripts
├── tests/               # Unit / integration tests (80%+ coverage)
├── .github/workflows/   # CI (lint, type-check, security, test, Docker) + CD
├── Dockerfile           # Multi-stage, non-root, healthchecked
├── docker-compose.yml
├── config.py            # pydantic-settings singleton
├── run.py
└── wsgi.py
```

## Local run

```bash
python -m pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env to set your provider and API keys
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
- use Gemini: `PROVIDER=google GOOGLE_API_KEY=your-key docker compose up --build`
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
| `POST` | `/api/agent/analyze` | Full agent analysis (parallel pipeline) |
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

CI pipeline (5 stages):

1. **Lint** — `ruff check` + `ruff format` + `flake8`
2. **Type-check** — `mypy` (non-blocking)
3. **Security** — `pip-audit` dependency scanning
4. **Test** — pytest on Python 3.11 + 3.12, coverage ≥80%
5. **Docker** — production build + health-endpoint smoke test

CD publishes multi-arch images to GHCR on `v*` tags.

## Useful commands

```bash
make lint          # Lint with flake8
make test          # Run tests
make test-cov      # Tests with coverage report
make ci            # Full CI: lint + test-cov
make docker-build  # Build production image
make docker-up     # Start with docker-compose
make docker-test   # Run tests in Docker
```

## Providers

| Provider | Config | Free? |
|----------|--------|-------|
| `rule_based` | No setup needed | ✅ |
| `ollama` | Install [Ollama](https://ollama.ai), `ollama pull llama3.2:3b` | ✅ |
| `google` | Set `GOOGLE_API_KEY` | Free tier |
| `openai` | Set `OPENAI_API_KEY` | Paid |
| `huggingface` | Set `HF_TOKEN` | Free tier |

## License

MIT
