"""DSPy Classification Studio — Flask MVC application.

Package layout (MVC + DDD):
    app/
    ├── domain/        # Enums, errors, value objects
    ├── models/        # Pydantic schemas, DSPy classifiers
    ├── views/         # Flask routes (HTTP layer)
    ├── controllers/   # Business logic orchestration
    ├── services/      # DSPy LM init, Knowledge Graph
    ├── agents/        # LangGraph pipelines
    ├── templates/     # Jinja2 + AngularJS HTML
    └── static/        # CSS, JS
"""

__version__ = "2.0.0"
