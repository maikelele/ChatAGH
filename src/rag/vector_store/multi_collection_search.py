from typing import Sequence

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from rag.embeddings.sentence_transformers_embeddings import SentenceTransformersEmbeddings
from rag.vector_store.qdrant_hybrid_search import QdrantHybridSearchVectorStore

QDRANT_COLLECTIONS = [
    "agh_edu_1",
    # "agh_edu_2",
    # "agh_edu_3",
    # "agh_edu_4",
    # "rekrutacja_sylabusy_agh",
    "miasteczko_akademik_agh",
    # "historia_agh",
    # "dss_agh"
]


class MultiCollectionSearch:
    """
    A search engine that performs multi-collection retrieval using Qdrant hybrid search.

    This class manages multiple Qdrant collections and executes parallel search queries
    across them. The results are reranked using cosine similarity based on dense embeddings.

    Attributes:
        collections (list): List of Qdrant collection names to search in.
        connectors (list): List of QdrantHybridSearchVectorStore instances for each collection.
        dense_embedding_model (SentenceTransformersEmbeddings): Model for generating dense embeddings.

    Methods:
        _search_single(connector, query, k_per_collection):
            Executes a search query on a single collection.

        _compute_similarity(query_embedding, result_embedding):
            Computes cosine similarity between query and result embeddings.

        search(query: str, k: int = 5, **kwargs):
            Searches across multiple collections in parallel, retrieves results,
            and reranks them using dense embedding similarity.

            Args:
                query (str): The search query.
                k (int): The number of top results to return.
                **kwargs: Optional keyword arguments, including 'k_per_collection'.

            Returns:
                list: A list of ranked search results, each containing text, similarity score,
                      and collection information.
    """
    def __init__(self):
        self.collections = QDRANT_COLLECTIONS
        self.connectors = [
            QdrantHybridSearchVectorStore(collection_name) for collection_name in self.collections
        ]
        self.dense_embedding_model = SentenceTransformersEmbeddings()

    def _search_single(self, connector, query, k_per_collection) -> Sequence:
        return connector.search(query, k_per_collection)

    def _compute_similarity(self, query_embedding, result_embedding) -> np.ndarray:
        return np.dot(query_embedding, result_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
        )

    def search(self, query: str, k: int = 5, **kwargs) -> Sequence:
        k_per_collection = min(kwargs.get("k_per_collection", k * 2), 20)

        all_results = []
        with ThreadPoolExecutor() as executor:
            future_to_connector = {
                executor.submit(self._search_single, connector, query, k_per_collection): connector
                for connector in self.connectors
            }
            for future in future_to_connector:
                collection_results = future.result()

                for result in collection_results:
                    result["collection"] = future_to_connector[future].collection_name
                all_results.extend(collection_results)

        if not all_results:
            return []

        query_dense_embedding = self.dense_embedding_model.embed([query])[0]

        # Rerank results using dense embeddings similarity
        for result in all_results:
            text = result.get("text", "")
            result_dense_embedding = self.dense_embedding_model.embed([text])[0]
            similarity = self._compute_similarity(query_dense_embedding, result_dense_embedding)
            result["similarity_score"] = float(similarity)

        ranked_results = sorted(all_results, key=lambda x: x.get("similarity_score", 0), reverse=True)

        return ranked_results[:k]
