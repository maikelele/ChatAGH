import uuid
from typing import List, Optional, Union

import chromadb
from chromadb.config import Settings

from langchain_core.documents import Document

from POC.rag.src.vector_stores.base_vector_store import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of the VectorStore."""

    def __init__(self, collection_name: str, embedding_function=None, persist_directory: Optional[str] = None):
        """
        Initialize a ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_function: Function to convert text to embeddings
            persist_directory: Optional directory to persist the database
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.embedding_function = embedding_function

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )

    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None):
        """
        Add documents to the ChromaDB collection.

        Args:
            documents: List of documents with at least "text" field
            embeddings: Optional pre-computed embeddings for the documents

        Returns:
            List of document IDs added to the store
        """
        doc_ids = []
        for doc in documents:
            doc_ids.append(str(uuid.uuid4()))

        texts = [doc.page_content for doc in documents]

        if embeddings is None and self.embedding_function:
            embeddings = self.embedding_function(texts)

        metadatas = [doc.metadata for doc in documents]
        for metadata in metadatas:
            for key in list(metadata.keys()):
                if metadata[key] == []:
                    del metadata[key]

        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return doc_ids

    def search(self, query: Union[str, List[float]], k: int = 5) -> List[Document]:
        """
        Search the ChromaDB collection.

        Args:
            query: Either a text query or a vector embedding
            k: Number of results to return

        Returns:
            List of documents with similarity scores
        """
        print(query)
        if isinstance(query, str) and self.embedding_function:
            query_embedding = self.embedding_function([query])[0]
        elif isinstance(query, list):
            query_embedding = query
        else:
            raise ValueError("Either provide a text query with an embedding function or a query embedding")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        docs = []
        for i in range(len(results["ids"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def delete(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the ChromaDB collection.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Success status
        """
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def get(self, document_ids: List[str]) -> List[Document]:
        """
        Retrieve documents by ID from the ChromaDB collection.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of retrieved documents
        """
        results = self.collection.get(
            ids=document_ids,
            include=["documents", "metadatas"]
        )

        docs = []
        for i in range(len(results["ids"])):
            text = results["documents"][i]
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
