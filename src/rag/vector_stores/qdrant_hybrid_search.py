import os

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding

from rag.embeddings.sentence_transformers_embeddings import SentenceTransformersEmbeddings


class QdrantHybridSearchVectorStore:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        self.dense_embedding_model = SentenceTransformersEmbeddings()
        self.sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        self.late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_embedding_model.dimension,
                        distance=models.Distance.COSINE,
                    ),
                    "late_interaction": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        )
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                }
            )
            
    def index(self, documents: list[Document]):
        for idx, doc in enumerate(documents):
            dense_embeddings = list(self.dense_embedding_model.embed([doc.page_content]))
            sparse_embeddings = list(self.sparse_embedding_model.passage_embed(doc.page_content))
            late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(doc.page_content))

            self.client.upload_points(
                self.collection_name,
                points=[
                    models.PointStruct(
                        id=idx,
                        vector={
                            "dense": dense_embeddings[0],
                            "sparse": sparse_embeddings[0].as_object(),
                            "late_interaction": late_interaction_embeddings[0].tolist(),
                        },
                        payload={
                            "url": doc.metadata.get("url"),
                            "title": doc.metadata.get("title"),
                            "description": doc.metadata.get("description"),
                            "date": doc.metadata.get("date"),
                            "tags": doc.metadata.get("tags"),
                            "source": doc.metadata.get("source"),
                            "text": doc.page_content,
                        }
                    )
                ],
            )
        
    def search(self, query: str, k: int = 5) -> list:
        dense_query_vector = self.dense_embedding_model.embed([query])[0]
        sparse_query_vector = next(self.sparse_embedding_model.query_embed(query))
        late_query_vector = next(self.late_interaction_embedding_model.query_embed(query))

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using="dense",
                limit=30,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using="sparse",
                limit=30,
            ),
        ]
        results = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=late_query_vector,
            using="late_interaction",
            with_payload=True,
            limit=k,
        )

        return [point.payload for point in results.points]