#!/usr/bin/env python3
"""
DSPy RAG Demo - Main Entry Point

This project demonstrates a complete RAG (Retrieval-Augmented Generation) system
using the latest DSPy framework with Google Gemini as the language model.

Features:
- Multiple RAG strategies (Basic, Advanced, Multi-hop)
- Vector storage with ChromaDB
- REST API with FastAPI
- Document management
- Summarization capabilities

Usage:
    # Run the API server
    python main.py serve

    # Run interactive demo
    python main.py demo

    # Run a quick test
    python main.py test
"""
import sys
import argparse
import dspy

from config import settings
from vector_store import VectorStore, create_retriever
from knowledge_base import get_sample_documents
from dspy_modules import BasicRAG, AdvancedRAG, MultiHopQA, DocumentSummarizer


def setup_dspy():
    """Configure DSPy with Google Gemini."""
    print(f"🔧 Configuring DSPy with Google Gemini ({settings.GOOGLE_MODEL})...")

    # DSPy 3.x with litellm uses "gemini/" prefix for Google models
    lm = dspy.LM(
        model=f"gemini/{settings.GOOGLE_MODEL}",
        api_key=settings.GOOGLE_API_KEY
    )
    dspy.configure(lm=lm)

    print("✅ DSPy configured successfully!")
    return lm


def setup_vector_store():
    """Initialize and populate the vector store."""
    print("📚 Setting up vector store...")

    vector_store = VectorStore(
        collection_name="knowledge_base",
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    # Load sample documents if empty
    if vector_store.count() == 0:
        print("   Loading sample documents...")
        docs = get_sample_documents()
        vector_store.add_documents(
            documents=[d["text"] for d in docs],
            ids=[d["id"] for d in docs],
            metadatas=[d["metadata"] for d in docs]
        )
        print(f"   ✅ Loaded {len(docs)} documents")
    else:
        print(f"   ✅ Found {vector_store.count()} existing documents")

    return vector_store


def run_demo():
    """Run an interactive demo of the DSPy RAG system."""
    print("\n" + "="*60)
    print("🚀 DSPy RAG Demo with Google Gemini")
    print("="*60 + "\n")

    # Setup
    setup_dspy()
    vector_store = setup_vector_store()
    retriever = create_retriever(vector_store, k=3)

    # Initialize modules
    basic_rag = BasicRAG(retriever)
    advanced_rag = AdvancedRAG(retriever, max_hops=2)
    summarizer = DocumentSummarizer()

    print("\n" + "-"*60)
    print("📝 Demo 1: Basic RAG Question Answering")
    print("-"*60)

    question = "What is DSPy and how does it work?"
    print(f"\n❓ Question: {question}\n")

    result = basic_rag(question=question)
    print(f"💡 Answer: {result.answer}\n")
    print(f"📖 Context used:\n{result.context[:500]}...")

    print("\n" + "-"*60)
    print("📝 Demo 2: Advanced RAG with Query Rewriting")
    print("-"*60)

    question2 = "How do DSPy optimizers improve LLM programs?"
    print(f"\n❓ Question: {question2}\n")

    result2 = advanced_rag(question=question2)
    print(f"🔍 Optimized Search Query: {result2.search_query}")
    print(f"💡 Answer: {result2.answer}")
    print(f"✅ Assessment: {result2.assessment}")
    print(f"📋 Reasoning: {result2.reasoning}")

    print("\n" + "-"*60)
    print("📝 Demo 3: Document Summarization")
    print("-"*60)

    doc = """
    Quantum computing leverages quantum mechanical phenomena like superposition
    and entanglement to process information. Unlike classical bits that are 0 or 1,
    quantum bits (qubits) can exist in multiple states simultaneously. This enables
    quantum computers to solve certain problems exponentially faster than classical
    computers, particularly in cryptography, drug discovery, and optimization.
    Major tech companies and startups are racing to build practical quantum computers.
    """
    print(f"\n📄 Document: {doc.strip()[:200]}...\n")

    summary_result = summarizer(document=doc)
    print(f"📋 Summary: {summary_result.summary}")

    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60 + "\n")


def run_interactive():
    """Run an interactive Q&A session."""
    print("\n" + "="*60)
    print("🤖 DSPy Interactive Q&A Session")
    print("="*60)
    print("Type 'quit' to exit, 'mode:basic/advanced/multihop' to change mode\n")

    # Setup
    setup_dspy()
    vector_store = setup_vector_store()
    retriever = create_retriever(vector_store, k=3)

    # Initialize modules
    modules = {
        "basic": BasicRAG(retriever),
        "advanced": AdvancedRAG(retriever, max_hops=2),
        "multihop": MultiHopQA(retriever, max_hops=3)
    }

    current_mode = "basic"

    while True:
        try:
            user_input = input(f"\n[{current_mode}] ❓ Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break

            if user_input.startswith("mode:"):
                new_mode = user_input.split(":")[1].strip()
                if new_mode in modules:
                    current_mode = new_mode
                    print(f"✅ Switched to {current_mode} mode")
                else:
                    print(f"❌ Invalid mode. Choose from: {list(modules.keys())}")
                continue

            # Process question
            print("\n🔄 Processing...")
            result = modules[current_mode](question=user_input)

            print(f"\n💡 Answer: {result.answer}")

            if hasattr(result, "search_query"):
                print(f"🔍 Search Query: {result.search_query}")
            if hasattr(result, "assessment"):
                print(f"✅ Assessment: {result.assessment}")
            if hasattr(result, "queries"):
                print(f"🔗 Queries Used: {result.queries}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_test():
    """Run a quick test to verify the setup."""
    print("\n🧪 Running quick test...")

    try:
        # Setup
        setup_dspy()
        vector_store = setup_vector_store()
        retriever = create_retriever(vector_store, k=3)

        # Test basic RAG
        basic_rag = BasicRAG(retriever)
        result = basic_rag(question="What is DSPy?")

        print(f"\n✅ Test passed!")
        print(f"   Question: What is DSPy?")
        print(f"   Answer preview: {result.answer[:200]}...")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


def run_server():
    """Run the FastAPI server."""
    import uvicorn

    print("\n🚀 Starting DSPy RAG API Server...")
    print(f"   Host: {settings.HOST}")
    print(f"   Port: {settings.PORT}")
    print(f"   Model: {settings.GOOGLE_MODEL}")
    print(f"   API Docs: http://{settings.HOST}:{settings.PORT}/docs\n")

    uvicorn.run(
        "api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DSPy RAG Demo with Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py serve      # Start the API server
    python main.py demo       # Run automated demo
    python main.py interactive # Interactive Q&A session
    python main.py test       # Quick test
        """
    )

    parser.add_argument(
        "command",
        choices=["serve", "demo", "interactive", "test"],
        nargs="?",
        default="demo",
        help="Command to run (default: demo)"
    )

    args = parser.parse_args()

    if args.command == "serve":
        run_server()
    elif args.command == "demo":
        run_demo()
    elif args.command == "interactive":
        run_interactive()
    elif args.command == "test":
        run_test()


if __name__ == "__main__":
    main()

