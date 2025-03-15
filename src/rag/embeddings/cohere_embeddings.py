import os
from typing import List

import cohere

from rag.embeddings.base_embeddings import BaseEmbeddings


class CohereEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "embed-multilingual-light-v3.0"):
        self.client = cohere.ClientV2(api_key=os.environ.get('COCOERE_API_KEY'))
        self.model = model_name

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_query",
            embedding_types=["float"],
        )
        return response.embeddings.float_



