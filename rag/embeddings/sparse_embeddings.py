import os
from typing import List, Dict, Any, Union, Callable
import pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class SparseEmbeddingStrategy:
    """
    Abstract base class for sparse embedding strategies
    """

    def query_embed(self, query: str) -> Dict[str, Union[List[int], List[float]]]:
        """
        Generate sparse embedding for a query

        Args:
            query (str): Input query

        Returns:
            Dict with indices and values of sparse embedding
        """
        raise NotImplementedError("Subclasses must implement query embedding")

    def passage_embed(self, passage: str) -> Dict[str, Union[List[int], List[float]]]:
        """
        Generate sparse embedding for a passage

        Args:
            passage (str): Input passage

        Returns:
            Dict with indices and values of sparse embedding
        """
        raise NotImplementedError("Subclasses must implement passage embedding")


import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class SPLADEEmbedding:
    """
    SPLADE sparse embedding implementation with robust error handling
    """

    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil"):
        """
        Initialize SPLADE embedding model

        Args:
            model_name (str): Hugging Face model name
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _get_sparse_embedding(self, text: str, is_query: bool = False):
        """
        Generate sparse embedding for text

        Args:
            text (str): Input text
            is_query (bool): Whether the text is a query

        Returns:
            Dict with sparse embedding indices and values
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Get model outputs
            outputs = self.model(**inputs)

            # Use last hidden state for creating sparse representation
            hidden_states = outputs.last_hidden_state.squeeze()

            # Create pseudo-logits by applying a transformation
            log_weights = torch.log1p(torch.relu(hidden_states))

            # Create sparse representation
            sparse_embeds = log_weights.sum(dim=0)

            # Find non-zero indices
            indices = torch.nonzero(sparse_embeds > 1e-4).squeeze().tolist()
            values = sparse_embeds[indices].tolist()

        return {
            'indices': indices,
            'values': values
        }

    def query_embed(self, query: str):
        """
        Generate sparse embedding for a query

        Args:
            query (str): Input query

        Returns:
            Dict with sparse embedding
        """
        return self._get_sparse_embedding(query, is_query=True)

    def passage_embed(self, passage: str):
        """
        Generate sparse embedding for a passage

        Args:
            passage (str): Input passage

        Returns:
            Dict with sparse embedding
        """
        return self._get_sparse_embedding(passage, is_query=False)


class TFIDFSparseEmbedding(SparseEmbeddingStrategy):
    """
    Simple TF-IDF based sparse embedding
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def query_embed(self, query: str):
        # Fit and transform just the query
        query_vector = self.vectorizer.fit_transform([query])
        indices = query_vector.indices.tolist()
        values = query_vector.data.tolist()
        return {'indices': indices, 'values': values}

    def passage_embed(self, passage: str):
        # Fit and transform the passage
        passage_vector = self.vectorizer.fit_transform([passage])
        indices = passage_vector.indices.tolist()
        values = passage_vector.data.tolist()
        return {'indices': indices, 'values': values}


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List


class CompressedTransformerSparseEmbedding:
    """
    Compressed Transformer-based Sparse Embedding

    Key Features:
    - Efficient dimension reduction
    - Transformer-based feature extraction
    - Sparse representation with controlled complexity
    """

    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 max_length: int = 512,
                 ):
        """
        Initialize compressed transformer sparse embedding

        Args:
            model_name (str): Transformer model name
            max_length (int): Maximum sequence length
            top_k (int): Number of top features to retain
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.max_length = max_length

    def _compress_embedding(self, embeddings: torch.Tensor) -> Dict[str, List[int]]:
        """
        Create a sparse embedding by:
        1. Applying importance scoring
        2. Selecting top-k features
        3. Normalizing selected features

        Args:
            embeddings (torch.Tensor): Input embeddings

        Returns:
            Dict with sparse embedding indices and values
        """
        # Compute feature importance via L2 norm across sequence
        feature_importance = torch.norm(embeddings, dim=0)
        # Select top-k most important features
        print(len(feature_importance))
        top_k_indices = torch.topk(feature_importance, k=26).indices

        # Extract and normalize top features
        compressed_features = embeddings[:, top_k_indices]
        compressed_features = F.normalize(compressed_features, p=2, dim=0)

        # Compute feature magnitude as values
        feature_values = torch.sum(torch.abs(compressed_features), dim=0)

        return {
            'indices': top_k_indices.tolist(),
            'values': feature_values.tolist()
        }

    def _get_sparse_embedding(self, text: str) -> Dict[str, List[int]]:
        """
        Generate sparse embedding for text

        Args:
            text (str): Input text

        Returns:
            Dict with sparse embedding indices and values
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )

            # Get model outputs
            outputs = self.model(**inputs)

            # Use last hidden state for sparse representation
            hidden_states = outputs.last_hidden_state.squeeze()

            # Compress embedding
            return self._compress_embedding(hidden_states.T)

    def query_embed(self, query: str) -> Dict[str, List[int]]:
        """
        Generate sparse embedding for a query

        Args:
            query (str): Input query

        Returns:
            Dict with sparse embedding
        """
        return self._get_sparse_embedding(query)

    def passage_embed(self, passage: str) -> Dict[str, List[int]]:
        """
        Generate sparse embedding for a passage

        Args:
            passage (str): Input passage

        Returns:
            Dict with sparse embedding
        """
        return self._get_sparse_embedding(passage)