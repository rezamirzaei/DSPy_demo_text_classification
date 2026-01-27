#!/usr/bin/env python3
"""
Quick test to verify the DSPy classification works.
Supports multiple providers: Ollama (local), Gemini, HuggingFace, OpenAI
Run this from PyCharm to test the setup.
"""
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env manually
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


def get_lm_config():
    """Get LM configuration based on PROVIDER env var."""
    provider = os.environ.get('PROVIDER', 'ollama').lower()

    if provider == 'ollama':
        # Ollama - FREE local models (Llama, Mistral, Phi, etc.)
        model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
        base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        return {
            'provider': 'ollama',
            'model': f"ollama/{model}",
            'api_base': base_url,
            'api_key': 'ollama',  # Ollama doesn't need a key
        }

    elif provider == 'gemini':
        # Google Gemini
        api_key = os.environ.get('GOOGLE_API_KEY', '')
        model = os.environ.get('GOOGLE_MODEL', 'gemini-2.0-flash')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set for Gemini provider")
        return {
            'provider': 'gemini',
            'model': f"gemini/{model}",
            'api_key': api_key,
        }

    elif provider == 'huggingface':
        # HuggingFace Inference API
        api_key = os.environ.get('HF_TOKEN', '')
        model = os.environ.get('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')
        if not api_key:
            raise ValueError("HF_TOKEN not set for HuggingFace provider")
        return {
            'provider': 'huggingface',
            'model': f"huggingface/{model}",
            'api_key': api_key,
        }

    elif provider == 'openai':
        # OpenAI
        api_key = os.environ.get('OPENAI_API_KEY', '')
        model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        return {
            'provider': 'openai',
            'model': model,
            'api_key': api_key,
        }

    else:
        raise ValueError(f"Unknown provider: {provider}. Use: ollama, gemini, huggingface, or openai")


def main():
    print("=" * 50)
    print("DSPy Classification Quick Test")
    print("=" * 50)

    # Get LM configuration
    try:
        config = get_lm_config()
        print(f"✅ Provider: {config['provider']}")
        print(f"✅ Model: {config['model']}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Import DSPy
    print("\n1. Importing DSPy...")
    try:
        import dspy
        print(f"   ✅ DSPy version: {getattr(dspy, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Run: pip install dspy")
        return

    # Configure DSPy
    print(f"\n2. Configuring DSPy with {config['provider']}...")
    try:
        lm_kwargs = {'model': config['model']}

        if 'api_key' in config:
            lm_kwargs['api_key'] = config['api_key']
        if 'api_base' in config:
            lm_kwargs['api_base'] = config['api_base']

        lm = dspy.LM(**lm_kwargs)
        dspy.configure(lm=lm)
        print("   ✅ DSPy configured!")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        if config['provider'] == 'ollama':
            print("\n   💡 To use Ollama (free local models):")
            print("      1. Install Ollama: https://ollama.ai")
            print("      2. Run: ollama pull llama3.2")
            print("      3. Ollama runs automatically on http://localhost:11434")
        return

    # Test sentiment classification
    print("\n3. Testing Sentiment Classification...")
    try:
        from app.models.classifier import SentimentClassifier

        classifier = SentimentClassifier()

        test_texts = [
            "I absolutely love this product! It's amazing!",
        ]

        for text in test_texts:
            print(f"\n   Text: \"{text[:50]}\"")
            result = classifier(text=text)
            print(f"   → Sentiment: {result.sentiment}")
            print(f"   → Confidence: {result.confidence}")
            reasoning = result.reasoning[:80] if len(result.reasoning) > 80 else result.reasoning
            print(f"   → Reasoning: {reasoning}...")

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
            print(f"   ❌ Rate limit exceeded!")
            print()
            print("   Your API quota is exhausted. Options:")
            print("   1. Install Ollama (FREE, no limits): https://ollama.ai")
            print("      Then run: ollama pull llama3.2")
            print("      And set PROVIDER=ollama in .env")
            print()
            print("   2. Get a new Gemini API key: https://aistudio.google.com/app/apikey")
            print()
            print("   3. Wait until tomorrow for daily quota reset")
        else:
            print(f"   ❌ Classification error: {e}")
            import traceback
            traceback.print_exc()
        return

    # Test topic classification
    print("\n4. Testing Topic Classification...")
    try:
        from app.models.classifier import TopicClassifier

        classifier = TopicClassifier()

        text = "Apple announced new iPhone features with improved AI capabilities for developers."
        print(f"\n   Text: \"{text[:50]}...\"")
        result = classifier(text=text)
        print(f"   → Topic: {result.topic}")
        print(f"   → Confidence: {result.confidence}")

    except Exception as e:
        print(f"   ❌ Classification error: {e}")
        return

    print("\n" + "=" * 50)
    print("✅ All tests passed! The application is ready.")
    print("=" * 50)
    print("\nTo start the web UI, run:")
    print("  python run.py --port 8000")
    print("\nThen open: http://localhost:8000")


if __name__ == "__main__":
    main()
