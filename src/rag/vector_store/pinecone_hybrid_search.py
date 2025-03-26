from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast
from collections import Counter
from tqdm.auto import tqdm
from typing import List, Dict, Union


class PineconeHybridSearchVectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dense_model_name: str = "intfloat/multilingual-e5-large",
        sparse_model_name: str = "bert-base-uncased"
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.dense_model = SentenceTransformer(dense_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(sparse_model_name)

    def indexing(
        self,
        documents: List[Document],
        batch_size: int = 32,
    ):

        contexts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        # Create Pinecone client
        pc = Pinecone(api_key=self.api_key)

        # Create index if not exists
        if self.index_name not in [idx.get("name") for idx in pc.list_indexes()]:
            pc.create_index(
                name=self.index_name,
                dimension=self.dense_model.get_sentence_embedding_dimension(),
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = pc.Index(self.index_name)

        # Generate vectors in batches
        for i in tqdm(range(0, len(contexts), batch_size)):
            i_end = min(i + batch_size, len(contexts))
            batch = contexts[i:i_end]

            # Generate dense vectors
            dense_embeds = self.dense_model.encode(batch).tolist()

            # Generate sparse vectors
            sparse_embeds = self._generate_sparse_vectors(batch)

            # Prepare metadata
            meta = metadata[i:i_end] if metadata else [{}] * len(batch)

            # Create vectors list
            vectors = []
            for idx, (context, dense, sparse, md) in enumerate(zip(contexts, dense_embeds, sparse_embeds, meta)):
                vectors.append({
                    "id": str(i + idx),
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": {
                        "url": md.get("url", ""),
                        "title": md.get("title", ""),
                        "content": context
                    }
                })

            # Upsert vectors
            index.upsert(vectors=vectors)

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> List[str]:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        # Create Pinecone client
        pc = Pinecone(api_key=self.api_key)

        # Generate sparse vector
        sparse_vec = self._generate_sparse_vectors([query])[0]

        # Generate dense vector
        dense_vec = self.dense_model.encode([query]).tolist()[0]

        # Scale vectors
        scaled_dense = [v * alpha for v in dense_vec]
        scaled_sparse = {
            "indices": sparse_vec["indices"],
            "values": [v * (1 - alpha) for v in sparse_vec["values"]]
        }

        index = pc.Index(self.index_name)
        response = index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            include_metadata=True
        )["matches"]

        response = [r["metadata"] for r in response]

        return response

    def _generate_sparse_vectors(
        self,
        texts: List[str]
    ) -> List[Dict[str, List]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512
        )["input_ids"]

        sparse_embeds = []
        for token_ids in inputs:
            counts = Counter(token_ids)
            indices = list(counts.keys())
            values = [float(v) for v in counts.values()]
            sparse_embeds.append({
                "indices": indices,
                "values": values
            })

        return sparse_embeds
