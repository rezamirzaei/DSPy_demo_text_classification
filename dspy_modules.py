"""
DSPy Modules for RAG-based Question Answering

This module contains DSPy signatures and modules for building
a retrieval-augmented generation (RAG) system.
"""
import dspy


# ============================================================
# DSPy Signatures - Define the input/output structure
# ============================================================

class GenerateAnswer(dspy.Signature):
    """Answer questions based on the provided context with detailed explanations."""

    context: str = dspy.InputField(desc="Relevant passages from the knowledge base")
    question: str = dspy.InputField(desc="User's question to answer")
    answer: str = dspy.OutputField(desc="Comprehensive answer based on the context")


class GenerateSearchQuery(dspy.Signature):
    """Generate an optimized search query to find relevant information."""

    question: str = dspy.InputField(desc="The original user question")
    search_query: str = dspy.OutputField(desc="Optimized search query for retrieval")


class AssessAnswer(dspy.Signature):
    """Assess if the answer is complete and accurate based on the context."""

    context: str = dspy.InputField(desc="The context used to generate the answer")
    question: str = dspy.InputField(desc="The original question")
    answer: str = dspy.InputField(desc="The generated answer to assess")
    assessment: str = dspy.OutputField(desc="Assessment: 'complete', 'partial', or 'insufficient'")
    reasoning: str = dspy.OutputField(desc="Explanation of the assessment")


class SummarizeDocument(dspy.Signature):
    """Summarize a document while preserving key information."""

    document: str = dspy.InputField(desc="The document to summarize")
    summary: str = dspy.OutputField(desc="Concise summary of the document")


# ============================================================
# DSPy Modules - Implement the logic
# ============================================================

class BasicRAG(dspy.Module):
    """Basic RAG module that retrieves context and generates answers."""

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        # Retrieve relevant context
        context = self.retriever(question)

        # Generate answer using the context
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            context=context,
            answer=prediction.answer
        )


class AdvancedRAG(dspy.Module):
    """
    Advanced RAG module with query rewriting and answer assessment.

    This module:
    1. Rewrites the query for better retrieval
    2. Retrieves relevant context
    3. Generates an answer
    4. Assesses the answer quality
    5. Optionally re-retrieves if assessment is insufficient
    """

    def __init__(self, retriever, max_hops: int = 2):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops

        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.assess_answer = dspy.ChainOfThought(AssessAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1: Generate optimized search query
        query_result = self.generate_query(question=question)
        search_query = query_result.search_query

        all_contexts = []
        combined_context = ""
        answer_result = None
        assessment_result = None

        for hop in range(self.max_hops):
            # Step 2: Retrieve context
            context = self.retriever(search_query if hop == 0 else question)
            all_contexts.append(context)

            # Combine all retrieved contexts
            combined_context = "\n\n---\n\n".join(all_contexts)

            # Step 3: Generate answer
            answer_result = self.generate_answer(
                context=combined_context,
                question=question
            )

            # Step 4: Assess the answer
            assessment_result = self.assess_answer(
                context=combined_context,
                question=question,
                answer=answer_result.answer
            )

            # If answer is complete, return it
            if assessment_result.assessment.lower() == "complete":
                break

        return dspy.Prediction(
            context=combined_context,
            search_query=search_query,
            answer=answer_result.answer,
            assessment=assessment_result.assessment,
            reasoning=assessment_result.reasoning
        )


class MultiHopQA(dspy.Module):
    """
    Multi-hop Question Answering module.

    Breaks down complex questions and iteratively gathers information.
    """

    def __init__(self, retriever, max_hops: int = 3):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops

        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context_parts = []
        queries_used = [question]
        combined_context = ""
        answer_result = None

        for hop in range(self.max_hops):
            # Generate search query based on what we know
            if hop == 0:
                current_query = question
            else:
                # Generate a follow-up query based on accumulated context
                query_result = self.generate_query(
                    question=f"Original question: {question}\n\nKnown information: {' '.join(context_parts)}\n\nWhat else do we need to find?"
                )
                current_query = query_result.search_query
                queries_used.append(current_query)

            # Retrieve context
            context = self.retriever(current_query)
            context_parts.append(context)

            # Try to answer with current context
            combined_context = "\n\n".join(context_parts)
            answer_result = self.generate_answer(
                context=combined_context,
                question=question
            )

        return dspy.Prediction(
            context=combined_context,
            queries=queries_used,
            answer=answer_result.answer
        )


class DocumentSummarizer(dspy.Module):
    """Summarize documents using DSPy."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(SummarizeDocument)

    def forward(self, document: str) -> dspy.Prediction:
        result = self.summarize(document=document)
        return dspy.Prediction(summary=result.summary)
