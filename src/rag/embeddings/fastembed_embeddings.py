from typing import List

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from rag.embeddings.base_embeddings import BaseEmbeddings

bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

class FastEmbedTextEmbedding(BaseEmbeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name)

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        dense_embeddings = list(self.model.passage_embed(texts))
        return [e.tolist() for e in dense_embeddings]
