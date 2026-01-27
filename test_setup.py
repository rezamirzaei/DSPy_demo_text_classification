#!/usr/bin/env python3
"""
Simple test script to verify DSPy installation and configuration.
Run this directly: python test_setup.py
"""
import sys

def main():
    print("=" * 50)
    print("DSPy Setup Verification")
    print("=" * 50)

    # Test 1: Check imports
    print("\n1. Checking imports...")
    try:
        import dspy
        print(f"   ✅ DSPy imported successfully (version: {getattr(dspy, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"   ❌ Failed to import DSPy: {e}")
        sys.exit(1)

    try:
        import chromadb
        print(f"   ✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import ChromaDB: {e}")
        sys.exit(1)

    try:
        from dotenv import load_dotenv
        print(f"   ✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import python-dotenv: {e}")
        sys.exit(1)

    # Test 2: Check configuration
    print("\n2. Checking configuration...")
    try:
        from config import settings
        if settings.GOOGLE_API_KEY:
            print(f"   ✅ GOOGLE_API_KEY is set (starts with: {settings.GOOGLE_API_KEY[:10]}...)")
        else:
            print("   ⚠️  GOOGLE_API_KEY is not set!")
        print(f"   📋 Model: {settings.GOOGLE_MODEL}")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        sys.exit(1)

    # Test 3: Check DSPy modules
    print("\n3. Checking DSPy modules...")
    try:
        from dspy_modules import BasicRAG, AdvancedRAG, DocumentSummarizer
        print("   ✅ Custom DSPy modules loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load DSPy modules: {e}")
        sys.exit(1)

    # Test 4: Check vector store
    print("\n4. Checking vector store...")
    try:
        from vector_store import VectorStore
        vs = VectorStore(collection_name="test_collection")
        print(f"   ✅ VectorStore works correctly")
    except Exception as e:
        print(f"   ❌ VectorStore error: {e}")
        sys.exit(1)

    # Test 5: Configure DSPy with Gemini
    print("\n5. Configuring DSPy with Google Gemini...")
    try:
        lm = dspy.LM(
            model=f"google/{settings.GOOGLE_MODEL}",
            api_key=settings.GOOGLE_API_KEY
        )
        dspy.configure(lm=lm)
        print("   ✅ DSPy configured with Gemini successfully")
    except Exception as e:
        print(f"   ❌ DSPy configuration error: {e}")
        sys.exit(1)

    # Test 6: Quick LLM test
    print("\n6. Testing LLM call...")
    try:
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is 2+2?")
        print(f"   ✅ LLM responded: {result.answer[:100]}...")
    except Exception as e:
        print(f"   ❌ LLM test failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ All tests passed! Your setup is ready.")
    print("=" * 50)
    print("\nYou can now run:")
    print("  python main.py demo        # Run the demo")
    print("  python main.py interactive # Interactive Q&A")
    print("  python main.py serve       # Start API server")


if __name__ == "__main__":
    main()
