from typing import List

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from langchain_core.documents import Document

from rag.src.embeddings.base_embeddings import BaseEmbeddings

DEFAULT_EMBEDDING_MODEL = 'ibm/granite-embedding-107m-multilingual'


class WatsonXEmbeddings(BaseEmbeddings):
    def __init__(self, client: APIClient, model: str = DEFAULT_EMBEDDING_MODEL):
        super().__init__()
        self.embeddings = Embeddings(
            api_client=client,
            model_id="ibm/slate-125m-english-rtrvr",
            params={
                "truncate_input_tokens": None
            },
        )
        self.model = model

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
