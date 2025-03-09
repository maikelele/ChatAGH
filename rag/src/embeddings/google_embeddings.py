import os
from typing import List

from google import genai
from langchain_core.documents import Document
from rag.src.embeddings.base_embeddings import BaseEmbeddings
from rag.src.utils.utils import retry_on_exception


class GoogleEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model_name

    def embed_documents(self, documents: List[Document], batch_size=None) -> List[List[float]]:
        texts = [doc.page_content for doc in documents]

        embed_vectors = []
        if batch_size is not None:
            for i in range(0, len(texts), batch_size):
                texts_to_embed = texts[i:i+batch_size]
                embedded_texts = self._embed(texts_to_embed)
                embed_vectors.extend(embedded_texts)
        else:
            embed_vectors = self._embed(texts)

        return embed_vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    @retry_on_exception(attempts=3, delay=5, backoff=3)
    def _embed(self, texts: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def __call__(self, *args, **kwargs):
        return self._embed(*args, **kwargs)


