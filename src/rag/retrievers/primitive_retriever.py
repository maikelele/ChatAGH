from typing import List

from langchain_core.documents import Document

from rag.retrievers.base_retriever import BaseRetriever
from rag.vector_stores.base_vector_store import BaseVectorStore
from rag.embeddings.base_embeddings import BaseEmbeddings


class PrimitiveRetriever(BaseRetriever):
    def __init__(self, vector_store: BaseVectorStore, number_of_chunks: int = 5):
        self.vector_store = vector_store
        self.number_of_chunks = number_of_chunks

    def invoke(self, query) -> List[Document]:
        retrieved_chunks = self.vector_store.search(query=query, k=self.number_of_chunks)
        return retrieved_chunks
