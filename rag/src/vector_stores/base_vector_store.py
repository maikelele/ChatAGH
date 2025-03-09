import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Abstract base class for vector stores used in ChatAGH"""

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None):
        """
        Add documents to the vector store.

        Args:
            documents: List of documents with at least "text" field
            embeddings: Optional pre-computed embeddings for the documents

        Returns:
            List of document IDs added to the store
        """
        pass

    @abstractmethod
    def search(self, query: Union[str, List[float]], k: int = 5) -> List[Document]:
        """
        Search the vector store for documents similar to the query.

        Args:
            query: Either a text query or a vector embedding
            k: Number of results to return

        Returns:
            List of documents with similarity scores
        """
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def get(self, document_ids: List[str]) -> List[Document]:
        """
        Retrieve documents by ID.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of retrieved documents
        """
        pass
