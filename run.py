#!/usr/bin/env python3
"""Application entry point."""

import argparse
import logging
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSPy Classification Studio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from config import get_settings
    settings = get_settings()

    logger.info(
        "Starting DSPy Classification Studio — provider=%s model=%s",
        settings.provider,
        settings.get_lm_config()["model"],
    )

    from app.controllers.classification_controller import ClassificationController
    controller = ClassificationController(settings)
    controller.initialize()

    from app.views.routes import create_app
    app = create_app(controller)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
