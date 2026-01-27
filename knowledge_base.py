"""
Sample knowledge base with documents about various topics.
Used to demonstrate the RAG system.
"""

SAMPLE_DOCUMENTS = [
    # Technology documents
    {
        "id": "tech_1",
        "text": """DSPy is a framework for programming with foundation models. Instead of 
        prompting language models with hand-crafted prompts, DSPy introduces programming 
        abstractions called signatures and modules. Signatures define the input/output 
        behavior of language model calls, while modules compose these calls into larger 
        programs. DSPy can automatically optimize prompts and weights using optimizers 
        like BootstrapFewShot and MIPRO. This makes LM programs more systematic, modular, 
        and maintainable compared to traditional prompt engineering.""",
        "metadata": {"topic": "technology", "subtopic": "AI frameworks"}
    },
    {
        "id": "tech_2",
        "text": """ChromaDB is an open-source embedding database designed for AI applications.
        It allows storing and querying vector embeddings efficiently. Key features include:
        automatic embedding generation, similarity search, metadata filtering, and persistent
        storage. ChromaDB integrates well with LangChain, LlamaIndex, and other AI frameworks.
        It supports multiple embedding functions and can run in-memory or with persistent storage.""",
        "metadata": {"topic": "technology", "subtopic": "databases"}
    },
    {
        "id": "tech_3",
        "text": """Google Gemini is a family of multimodal AI models developed by Google DeepMind.
        The Gemini family includes Ultra, Pro, and Flash variants optimized for different use cases.
        Gemini models can process text, images, audio, and video. They excel at reasoning tasks,
        code generation, and multilingual capabilities. The API is accessible through Google AI Studio
        and Vertex AI, with competitive pricing for developers.""",
        "metadata": {"topic": "technology", "subtopic": "AI models"}
    },

    # Science documents
    {
        "id": "science_1",
        "text": """Quantum computing leverages quantum mechanical phenomena like superposition
        and entanglement to process information. Unlike classical bits that are 0 or 1,
        quantum bits (qubits) can exist in multiple states simultaneously. This enables
        quantum computers to solve certain problems exponentially faster than classical
        computers, particularly in cryptography, drug discovery, and optimization.""",
        "metadata": {"topic": "science", "subtopic": "quantum physics"}
    },
    {
        "id": "science_2",
        "text": """CRISPR-Cas9 is a revolutionary gene-editing technology that allows precise
        modifications to DNA sequences. Discovered in bacteria as a defense mechanism against
        viruses, CRISPR has been adapted for editing genes in plants, animals, and humans.
        Applications include treating genetic diseases, developing disease-resistant crops,
        and creating animal models for research. Ethical considerations remain important.""",
        "metadata": {"topic": "science", "subtopic": "biology"}
    },

    # Business documents
    {
        "id": "business_1",
        "text": """Retrieval-Augmented Generation (RAG) is transforming enterprise AI applications.
        By combining large language models with external knowledge bases, RAG systems provide
        more accurate and up-to-date responses. Key benefits include: reduced hallucinations,
        ability to cite sources, easy knowledge updates without retraining, and compliance
        with data governance requirements. Major companies use RAG for customer support,
        internal knowledge management, and document analysis.""",
        "metadata": {"topic": "business", "subtopic": "AI applications"}
    },
    {
        "id": "business_2",
        "text": """The AI infrastructure market is experiencing rapid growth, with demand for
        GPUs, specialized AI chips, and cloud services skyrocketing. NVIDIA dominates the
        GPU market for AI training, while companies like AMD, Intel, and startups develop
        competing solutions. Cloud providers (AWS, Azure, GCP) offer AI-specific services
        and infrastructure. Edge AI deployment is growing for latency-sensitive applications.""",
        "metadata": {"topic": "business", "subtopic": "AI market"}
    },

    # Programming documents
    {
        "id": "prog_1",
        "text": """Python remains the dominant language for AI and machine learning development.
        Key libraries include: NumPy for numerical computing, Pandas for data manipulation,
        Scikit-learn for traditional ML, PyTorch and TensorFlow for deep learning, and
        Hugging Face Transformers for NLP. The Python ecosystem's simplicity and extensive
        library support make it ideal for rapid prototyping and production deployment.""",
        "metadata": {"topic": "programming", "subtopic": "Python"}
    },
    {
        "id": "prog_2",
        "text": """FastAPI is a modern Python web framework for building APIs. It's built on
        Starlette for web handling and Pydantic for data validation. Key features include:
        automatic API documentation with OpenAPI/Swagger, async support, type hints,
        dependency injection, and excellent performance. FastAPI is popular for building
        ML model serving endpoints and microservices.""",
        "metadata": {"topic": "programming", "subtopic": "web frameworks"}
    },

    # Additional DSPy-specific documents
    {
        "id": "dspy_1",
        "text": """DSPy Signatures are declarative specifications that define the semantic
        behavior of language model calls. A signature specifies input fields and output fields
        with descriptions. For example, a question-answering signature might have 'context'
        and 'question' as inputs and 'answer' as output. Signatures are compiled into optimized
        prompts during the optimization phase.""",
        "metadata": {"topic": "technology", "subtopic": "DSPy"}
    },
    {
        "id": "dspy_2",
        "text": """DSPy Modules are composable building blocks that process inputs and produce
        outputs using language models. Built-in modules include: dspy.Predict (basic LM call),
        dspy.ChainOfThought (reasoning with intermediate steps), dspy.ReAct (reasoning with
        tool use), and dspy.ProgramOfThought (code generation for reasoning). Custom modules
        can combine these primitives for complex pipelines.""",
        "metadata": {"topic": "technology", "subtopic": "DSPy"}
    },
    {
        "id": "dspy_3",
        "text": """DSPy optimizers automatically improve program performance by optimizing
        prompts, few-shot examples, and module weights. BootstrapFewShot generates training
        examples from a few labeled examples. MIPRO uses Bayesian optimization to find
        optimal instruction prompts. MIPROv2 combines instruction and few-shot optimization.
        Optimizers require a metric function to evaluate program outputs.""",
        "metadata": {"topic": "technology", "subtopic": "DSPy"}
    }
]


def get_sample_documents():
    """Return sample documents for indexing."""
    return SAMPLE_DOCUMENTS


def get_documents_by_topic(topic: str):
    """Filter documents by topic."""
    return [doc for doc in SAMPLE_DOCUMENTS if doc["metadata"]["topic"] == topic]
