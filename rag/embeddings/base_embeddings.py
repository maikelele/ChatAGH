from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """
    Abstract base class for embedding models.
    Defines the interface for all embedding implementations.
    """
    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        pass

    @property
    def dimension(self) -> int:
        return len(self.embed(["test"])[0])