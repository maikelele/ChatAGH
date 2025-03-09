from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseModel(ABC):
    """
    Abstract base class for inference with LLM.

    This class defines the interface for generating responses from a language model,
    either from a given question alone or with additional context from documents.
    """

    @abstractmethod
    def generate(self, question: str):
        """
        Generate a response to a given question.

        Args:
            question (str): The input question for the model.

        Returns:
            Any: The generated response.
        """
        pass

    @abstractmethod
    def generate_from_documents(self, question: str, documents: list[Document]):
        """
        Generate a response to a given question using additional context from documents.

        Args:
            question (str): The input question for the model.
            documents (list[Document]): A list of documents providing context.

        Returns:
            Any: The generated response.
        """
        pass
