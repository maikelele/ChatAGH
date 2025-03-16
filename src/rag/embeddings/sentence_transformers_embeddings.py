from typing import List

from sentence_transformers import SentenceTransformer

from rag.embeddings.base_embeddings import BaseEmbeddings


class SentenceTransformersEmbeddings(BaseEmbeddings):
    """
    A class for generating dense text embeddings using SentenceTransformer models.

    This class utilizes the SentenceTransformer library to encode a list of text strings
    into their corresponding vector representations. The output is formatted as a list of lists of floats.

    Attributes:
        model (SentenceTransformer): The SentenceTransformer model instance used for encoding text.

    Methods:
        embed(texts: List[str], **kwargs) -> List[List[float]]:
            Encodes a list of text strings into dense vector representations.
    """
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v1"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.model.encode(texts).tolist()
