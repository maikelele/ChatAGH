from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseChunker(ABC):
    """
    Abstract base class for document chunking.
    Defines the interface for all chunkers implementations.
    """

    @abstractmethod
    def chunk(self, documents: List[Any]) -> List[Any]:
        """
        Split input documents into chunks.

        Args:
            documents: List of documents to be chunked

        Returns:
            List of document chunks
        """
        pass

    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Split a text string into chunks.

        Args:
            text: Text to be chunked
            metadata: Optional metadata to include with each chunk

        Returns:
            List of text chunks
        """
        pass


