from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseEmbeddings(ABC):
    """
    Abstract base class for embedding models.
    Defines the interface for all embedding implementations.
    """
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of documents to embed

        Returns:
            List of embedding vectors, one per document
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass