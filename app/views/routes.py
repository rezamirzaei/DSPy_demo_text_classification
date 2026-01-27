"""
Flask Routes (Views)

Defines all HTTP routes for the classification application.
This is the View layer in MVC architecture.
"""
from flask import Blueprint, render_template, request, jsonify, current_app
from app.models.schemas import ClassificationRequest, ClassifierType
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    """Render the main classification UI."""
    return render_template('index.html')


@bp.route('/health')
def health():
    """Health check endpoint."""
    controller = current_app.config.get('controller')
    return jsonify({
        'status': 'healthy' if controller and controller.is_initialized else 'not ready',
        'model': current_app.config.get('MODEL', 'unknown'),
        'classifiers': controller.get_available_classifiers() if controller else []
    })


@bp.route('/api/classify', methods=['POST'])
def classify():
    """
    API endpoint for text classification.

    Request JSON:
    {
        "text": "Text to classify",
        "classifier_type": "sentiment|topic|intent",
        "categories": ["optional", "custom", "categories"],
        "intents": ["optional", "custom", "intents"]
    }
    """
    controller = current_app.config.get('controller')

    if not controller or not controller.is_initialized:
        return jsonify({
            'success': False,
            'error': 'Classification service not initialized'
        }), 503

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        # Create classification request
        classifier_type = data.get('classifier_type', 'sentiment')

        try:
            classifier_type_enum = ClassifierType(classifier_type)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid classifier_type. Must be one of: {[e.value for e in ClassifierType]}'
            }), 400

        classification_request = ClassificationRequest(
            text=data['text'],
            classifier_type=classifier_type_enum,
            categories=data.get('categories'),
            intents=data.get('intents')
        )

        # Perform classification
        response = controller.classify(classification_request)

        return jsonify({
            'success': response.success,
            'text': response.text,
            'classifier_type': response.classifier_type,
            'result': response.result,
            'error': response.error
        })

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/classify/batch', methods=['POST'])
def batch_classify():
    """
    API endpoint for batch classification.

    Request JSON:
    {
        "texts": ["text1", "text2", ...],
        "classifier_type": "sentiment|topic|intent"
    }
    """
    controller = current_app.config.get('controller')

    if not controller or not controller.is_initialized:
        return jsonify({
            'success': False,
            'error': 'Classification service not initialized'
        }), 503

    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: texts'
            }), 400

        from app.models.schemas import BatchClassificationRequest

        batch_request = BatchClassificationRequest(
            texts=data['texts'],
            classifier_type=ClassifierType(data.get('classifier_type', 'sentiment')),
            categories=data.get('categories'),
            intents=data.get('intents')
        )

        response = controller.batch_classify(batch_request)

        return jsonify({
            'success': True,
            'total': response.total,
            'successful': response.successful,
            'failed': response.failed,
            'results': [r.model_dump() for r in response.results]
        })

    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/api/classifiers')
def list_classifiers():
    """List available classifiers."""
    controller = current_app.config.get('controller')

    classifiers_info = {
        'sentiment': {
            'name': 'Sentiment Analysis',
            'description': 'Classify text as positive, negative, or neutral',
            'outputs': ['sentiment', 'confidence', 'reasoning']
        },
        'topic': {
            'name': 'Topic Classification',
            'description': 'Categorize text into topics/categories',
            'outputs': ['topic', 'confidence', 'reasoning'],
            'default_categories': [
                'Technology', 'Business', 'Science', 'Health',
                'Sports', 'Entertainment', 'Politics', 'Education', 'Other'
            ]
        },
        'intent': {
            'name': 'Intent Detection',
            'description': 'Detect user intent and extract entities',
            'outputs': ['intent', 'confidence', 'entities', 'reasoning'],
            'default_intents': [
                'question', 'command', 'greeting', 'complaint',
                'feedback', 'request', 'information', 'other'
            ]
        }
    }

    return jsonify({
        'available': controller.get_available_classifiers() if controller else [],
        'details': classifiers_info
    })


def create_app(controller):
    """
    Create and configure the Flask application.

    Args:
        controller: ClassificationController instance

    Returns:
        Configured Flask app
    """
    from flask import Flask

    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static'
    )

    app.config['controller'] = controller
    app.config['MODEL'] = controller.model if controller else 'unknown'

    app.register_blueprint(bp)

    return app
