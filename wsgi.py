"""WSGI entrypoint for Gunicorn."""

from app.controllers.classification_controller import ClassificationController
from app.views.routes import create_app
from config import get_settings

settings = get_settings()
controller = ClassificationController(settings)
controller.initialize()
app = create_app(controller)
