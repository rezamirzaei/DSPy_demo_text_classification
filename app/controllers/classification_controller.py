"""
Classification Controller

Handles the business logic for text classification operations.
Acts as the bridge between Views and Models in MVC architecture.
Supports multiple LLM providers: Ollama (local), Gemini, HuggingFace, OpenAI
"""
import dspy
import os
from typing import List, Optional, Dict, Any
import logging

from app.models.classifier import (
    SentimentClassifier,
    TopicClassifier,
    IntentClassifier
)
from app.models.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse
)

logger = logging.getLogger(__name__)


def get_lm_config():
    """Get LM configuration based on PROVIDER env var."""
    provider = os.environ.get('PROVIDER', 'ollama').lower()

    if provider == 'ollama':
        model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
        base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        return {
            'provider': 'ollama',
            'model': f"ollama/{model}",
            'api_base': base_url,
            'api_key': 'ollama',
        }

    elif provider == 'gemini':
        api_key = os.environ.get('GOOGLE_API_KEY', '')
        model = os.environ.get('GOOGLE_MODEL', 'gemini-2.0-flash')
        return {
            'provider': 'gemini',
            'model': f"gemini/{model}",
            'api_key': api_key,
        }

    elif provider == 'huggingface':
        api_key = os.environ.get('HF_TOKEN', '')
        model = os.environ.get('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')
        return {
            'provider': 'huggingface',
            'model': f"huggingface/{model}",
            'api_key': api_key,
        }

    elif provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY', '')
        model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        return {
            'provider': 'openai',
            'model': model,
            'api_key': api_key,
        }

    else:
        # Default fallback to ollama
        return {
            'provider': 'ollama',
            'model': 'ollama/llama3.2',
            'api_base': 'http://localhost:11434',
            'api_key': 'ollama',
        }


class ClassificationController:
    """
    Controller for managing text classification operations.

    This controller:
    - Initializes and manages DSPy classifiers
    - Handles classification requests
    - Manages classifier lifecycle
    - Supports multiple LLM providers
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the classification controller.

        Args:
            api_key: API key (optional, will use env vars)
            model: Model name (optional, will use env vars)
        """
        self.api_key = api_key
        self.model = model
        self._initialized = False
        self._classifiers: Dict[str, Any] = {}
        self._provider = None

    def initialize(self) -> bool:
        """
        Initialize DSPy and classifiers.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Get LM config from environment
            config = get_lm_config()
            self._provider = config['provider']
            self.model = config['model']

            logger.info(f"Initializing DSPy with provider: {self._provider}, model: {self.model}")

            # Build LM kwargs
            lm_kwargs = {'model': config['model']}

            if config.get('api_key'):
                lm_kwargs['api_key'] = config['api_key']
            if config.get('api_base'):
                lm_kwargs['api_base'] = config['api_base']

            # Configure DSPy
            lm = dspy.LM(**lm_kwargs)
            dspy.configure(lm=lm)

            # Initialize classifiers
            self._classifiers = {
                'sentiment': SentimentClassifier(),
                'topic': TopicClassifier(),
                'intent': IntentClassifier()
            }

            self._initialized = True
            logger.info(f"Classification controller initialized successfully with {self._provider}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize classification controller: {e}")
            self._initialized = False
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._initialized

    def get_available_classifiers(self) -> List[str]:
        """Get list of available classifier types."""
        return list(self._classifiers.keys())

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """
        Perform classification on the given text.

        Args:
            request: Classification request with text and classifier type

        Returns:
            Classification response with results
        """
        if not self._initialized:
            return ClassificationResponse(
                text=request.text,
                classifier_type=request.classifier_type.value,
                result={},
                success=False,
                error="Controller not initialized. Call initialize() first."
            )

        try:
            classifier_type = request.classifier_type.value
            classifier = self._classifiers.get(classifier_type)

            if not classifier:
                return ClassificationResponse(
                    text=request.text,
                    classifier_type=classifier_type,
                    result={},
                    success=False,
                    error=f"Unknown classifier type: {classifier_type}"
                )

            # Perform classification based on type
            if classifier_type == 'sentiment':
                result = classifier(text=request.text)
                result_dict = {
                    'sentiment': result.sentiment,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning
                }

            elif classifier_type == 'topic':
                result = classifier(
                    text=request.text,
                    categories=request.categories
                )
                result_dict = {
                    'topic': result.topic,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'available_categories': result.available_categories
                }

            elif classifier_type == 'intent':
                result = classifier(
                    text=request.text,
                    intents=request.intents
                )
                result_dict = {
                    'intent': result.intent,
                    'confidence': result.confidence,
                    'entities': result.entities,
                    'reasoning': result.reasoning,
                    'available_intents': result.available_intents
                }
            else:
                result_dict = {}

            return ClassificationResponse(
                text=request.text,
                classifier_type=classifier_type,
                result=result_dict,
                success=True
            )

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResponse(
                text=request.text,
                classifier_type=request.classifier_type.value,
                result={},
                success=False,
                error=str(e)
            )

    def classify_sentiment(self, text: str) -> Dict[str, Any]:
        """Shortcut for sentiment classification."""
        request = ClassificationRequest(text=text, classifier_type="sentiment")
        response = self.classify(request)
        return response.result if response.success else {"error": response.error}

    def classify_topic(self, text: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Shortcut for topic classification."""
        request = ClassificationRequest(
            text=text,
            classifier_type="topic",
            categories=categories
        )
        response = self.classify(request)
        return response.result if response.success else {"error": response.error}

    def classify_intent(self, text: str, intents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Shortcut for intent classification."""
        request = ClassificationRequest(
            text=text,
            classifier_type="intent",
            intents=intents
        )
        response = self.classify(request)
        return response.result if response.success else {"error": response.error}

    def batch_classify(self, request: BatchClassificationRequest) -> BatchClassificationResponse:
        """
        Perform batch classification on multiple texts.

        Args:
            request: Batch classification request

        Returns:
            Batch classification response with all results
        """
        results = []
        successful = 0
        failed = 0

        for text in request.texts:
            single_request = ClassificationRequest(
                text=text,
                classifier_type=request.classifier_type,
                categories=request.categories,
                intents=request.intents
            )
            response = self.classify(single_request)
            results.append(response)

            if response.success:
                successful += 1
            else:
                failed += 1

        return BatchClassificationResponse(
            results=results,
            total=len(request.texts),
            successful=successful,
            failed=failed
        )
