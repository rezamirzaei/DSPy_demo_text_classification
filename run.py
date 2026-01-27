#!/usr/bin/env python3
"""
DSPy Text Classification Application - Main Runner

MVC Architecture:
- Models: DSPy classifiers (app/models/)
- Views: Flask routes & templates (app/views/, app/templates/)
- Controllers: Business logic (app/controllers/)

Usage:
    python run.py              # Start the web server
    python run.py --port 5000  # Custom port
    python run.py --debug      # Debug mode
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to load dotenv, but don't fail if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Load .env manually if dotenv not available
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DSPy Text Classification Web Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Start server on port 8000
    python run.py --port 5000        # Start on port 5000
    python run.py --debug            # Enable debug mode
    python run.py --host 127.0.0.1   # Localhost only

Providers (set PROVIDER in .env):
    ollama      - FREE local models (default)
    gemini      - Google Gemini (requires GOOGLE_API_KEY)
    huggingface - HuggingFace (requires HF_TOKEN)
    openai      - OpenAI (requires OPENAI_API_KEY)
        """
    )

    parser.add_argument(
        '--host',
        default=os.getenv('HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('PORT', '8000')),
        help='Port to listen on (default: 8000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=os.getenv('DEBUG', 'false').lower() == 'true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.debug)

    # Get provider info
    provider = os.getenv('PROVIDER', 'ollama')

    logger.info("=" * 60)
    logger.info("🧠 DSPy Text Classification Application")
    logger.info("=" * 60)
    logger.info(f"Provider: {provider}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug: {args.debug}")
    logger.info("=" * 60)

    # Initialize controller (it will read provider settings from env)
    logger.info("Initializing classification controller...")

    from app.controllers import ClassificationController
    controller = ClassificationController()

    if not controller.initialize():
        logger.error("Failed to initialize classification controller!")
        if provider == 'ollama':
            logger.info("\n💡 To use Ollama (free local models):")
            logger.info("   1. Install Ollama: https://ollama.ai")
            logger.info("   2. Run: ollama pull llama3.2")
            logger.info("   3. Ollama runs automatically on http://localhost:11434")
        sys.exit(1)

    logger.info(f"✅ Controller initialized with model: {controller.model}")
    logger.info(f"Available classifiers: {controller.get_available_classifiers()}")

    # Create Flask app
    from app.views import create_app
    app = create_app(controller)

    # Run server
    logger.info(f"\n🚀 Starting server at http://{args.host}:{args.port}")
    logger.info(f"📱 Open http://localhost:{args.port} in your browser\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
