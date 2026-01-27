"""
Vector Store implementation using ChromaDB for document retrieval.
"""
import os
from typing import List, Optional
import chromadb


class VectorStore:
    """
    ChromaDB-based vector store for document storage and retrieval.

    This class provides:
    - Document indexing with embeddings
    - Semantic search capabilities
    - Persistent storage
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            ids: Optional list of unique IDs for documents
            metadatas: Optional list of metadata dicts for documents
        """
        if ids is None:
            # Generate IDs based on existing count
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(
        self,
        query: str,
        n_results: int = 3
    ) -> List[str]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of relevant document texts
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Extract documents from results
        if results and results["documents"]:
            return results["documents"][0]
        return []

    def search_with_metadata(
        self,
        query: str,
        n_results: int = 3
    ) -> dict:
        """
        Search for relevant documents with metadata.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dict containing documents, metadatas, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate the collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


def create_retriever(vector_store: VectorStore, k: int = 3):
    """
    Create a retriever function compatible with DSPy modules.

    Args:
        vector_store: VectorStore instance
        k: Number of documents to retrieve

    Returns:
        A callable that takes a query and returns context string
    """
    def retriever(query: str) -> str:
        documents = vector_store.search(query, n_results=k)
        return "\n\n".join(documents) if documents else "No relevant context found."

    return retriever
