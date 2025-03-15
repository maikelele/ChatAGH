import os
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag.embeddings.base_embeddings import BaseEmbeddings
from rag.utils.utils import retry_on_exception


class GoogleEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.model = model_name

    def embed_documents(self, documents: List[Document] | List[str], batch_size=None) -> List[List[float]]:
        if isinstance(documents[0], str):
            texts = documents
        else:
            texts = [doc.page_content for doc in documents]

        return self.embeddings.embed_documents(texts, batch_size=batch_size)

    @retry_on_exception(attempts=3)
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


