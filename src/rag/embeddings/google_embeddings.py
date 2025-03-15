import os
from typing import List

from google import genai
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag.embeddings.base_embeddings import BaseEmbeddings
from rag.utils.utils import retry_on_exception


class GoogleEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        # self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.model = model_name

    def embed_documents(self, documents: List[Document] | List[str], batch_size=None) -> List[List[float]]:
        if isinstance(documents[0], str):
            texts = documents
        else:
            texts = [doc.page_content for doc in documents]

        return self.embeddings.embed_documents(texts, batch_size=batch_size)

        # embed_vectors = []
        # if batch_size is not None:
        #     for i in range(0, len(texts), batch_size):
        #         print(f"EMBEDDING {i}/{len(texts)}")
        #         texts_to_embed = texts[i:i+batch_size]
        #         embedded_texts = self._embed(texts_to_embed)
        #         embed_vectors.extend(embedded_texts)
        # else:
        #     embed_vectors = self._embed(texts)
        #
        # return embed_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
        # return self._embed([text])[0]

    @retry_on_exception(attempts=3, delay=5, backoff=3)
    def _embed(self, texts: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def __call__(self, *args, **kwargs):
        return self._embed(*args, **kwargs)[0]

