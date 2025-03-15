from typing import List

from sentence_transformers import SentenceTransformer

from rag.embeddings.base_embeddings import BaseEmbeddings


class SentenceTransformersEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v1"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.model.encode(texts).tolist()