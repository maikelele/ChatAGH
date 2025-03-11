import uuid
from typing import List, Optional, Union, Callable

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.rag.vector_stores.base_vector_store import BaseVectorStore

class QuadrantCloudVectorStore(BaseVectorStore):
    def __init__(
        self,
        api_key: str,
        collection_name: str,
        url: str,
        embedding_fn: Optional[Callable[[str], List[float]]],
        vector_size: int = 768,
        distance_metric: str = "Cosine",
    ):
        """
        Initialize the Qdrant Cloud vector store.

        Args:
            api_key: Your Qdrant API key.
            collection_name: The collection name in Qdrant.
            embedding_fn: Optional function to convert text to vector embeddings.
            vector_size: The size of the vectors to be stored.
            distance_metric: The distance metric to be used ("Cosine", "Euclidean", etc.).
            url: URL for your Qdrant Cloud instance.
        """
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.client = QdrantClient(url=url, api_key=api_key)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Check if the collection exists; if not, create it.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=getattr(rest.Distance, self.distance_metric.upper())
                )
            )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents along with their embeddings to the Qdrant collection.

        If embeddings are not provided, the embedding_fn (if set) is used to compute them.
        Each document is assigned a unique ID.

        Args:
            documents: List of Document objects.
            embeddings: Optional list of pre-computed embeddings.

        Returns:
            List of document IDs added.
        """
        points = []
        doc_ids = []

        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError("Length of embeddings must match the number of documents")

        for idx, doc in enumerate(documents):
            print(f"Adding document: {idx} / {len(documents)}")

            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)

            if embeddings is not None:
                vector = embeddings[idx]
            elif self.embedding_fn is not None:
                vector = self.embedding_fn(doc.text)
            else:
                raise ValueError("No embeddings provided and no embedding function available")

            payload = {"text": doc.page_content}
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                payload.update(doc.metadata)

            point = rest.PointStruct(
                id=doc_id,
                vector=vector,
                payload=payload
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return doc_ids

    def search(self, query: Union[str, List[float]], k: int = 5) -> List[Document]:
        """
        Search the Qdrant collection for documents similar to the query.

        If the query is a string, it is converted to an embedding using the embedding_fn.
        The resulting hits are converted back into Document objects with scores attached to the metadata.

        Args:
            query: Either a text query or a vector embedding.
            k: Number of results to return.

        Returns:
            List of Document objects enriched with similarity scores.
        """
        if isinstance(query, str):
            if self.embedding_fn is None:
                raise ValueError("Text query provided but no embedding function available")
            query_vector = self.embedding_fn(query)
        else:
            query_vector = query

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k
        )

        results = []
        for point in search_result.points:
            page_content = point.payload["text"]
            metadata = {k: v for k, v in point.payload.items() if k != "text"}
            results.append(Document(page_content=page_content, metadata=metadata))

        return results
