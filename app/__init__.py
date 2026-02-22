"""DSPy Classification Studio — Flask MVC application.

Package layout (MVC + DDD):
    app/
    ├── domain/        # Enums, errors, value objects
    ├── models/        # Pydantic schemas, DSPy classifiers, BootstrapFewShot optimizer
    ├── views/         # Flask routes (HTTP layer)
    ├── controllers/   # Business logic orchestration
    ├── services/      # DSPy LM init, Knowledge Graph, hybrid analysis engines
    ├── agents/        # LangGraph pipelines (parallel fan-out)
    ├── templates/     # Jinja2 + AngularJS HTML
    └── static/        # CSS, JS
"""

__version__ = "2.1.0"
