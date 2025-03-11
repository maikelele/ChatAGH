from typing import List
from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseRetriever(ABC):
    @abstractmethod
    def invoke(self, query) -> List[Document]:
        pass
