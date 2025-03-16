# Contents
- [Overview](#overview)
- [Developer guide](#developer-guide)
- [Data sources and preparation](#data-sources-and-preparation)
- [RAG implementation](#rag-implementation)
- [Metrics and evaluation](#metrics-and-evaluation)
- [Future Improvements](#future-improvements)

# Overview
Chat AGH is a Retrieval-Augmented Generation (RAG) system designed to deliver accurate and relevant information about academic matters at AGH University of Science and Technology. The system aggregates data sourced from university websites, enabling it to provide comprehensive answers to a wide range of user inquiries, which may include topics related to admissions, faculty-specific information, campus facilities, events, etc.

# Developer guide

### Clone repository
```
git clone https://github.com/witoldnowogorski/ChatAGH
cd ChatAGH
```

### Create environment
```
python3 -m venv chat_agh
```
Unix / MacOS
```
source chat_agh/bin/activate
```
Windows
```
cd chat_agh
.\Scripts\activate
```

### Install requirements
```
pip install -r requirements.txt
```

### Credentials
Add `.env` file with your credentials in config directory, you can find required credentials in `.env.template`

### Run streamlit app
```
streamlit run src/streamlit_app.py
```

# Data sources and preparation

# RAG implementation

## Indexing
Designed to enable efficient retrieval of text documents using a hybrid search approach. The architecture integrates multiple embedding techniques and a scalable vector store for optimized information retrieval.


#### Vector Store

The project utilizes Qdrant, a high-performance vector database optimized for hybrid search. Qdrant enables efficient storage and retrieval of vector representations, supporting multiple embedding types.

#### Hybrid Search

To enhance search quality, the system employs a hybrid retrieval mechanism combining:
- Dense Embeddings – Derived from `distiluse-base-multilingual-cased-v1`, a transformer-based sentence embedding model optimized for semantic similarity.
- Sparse Embeddings – Generated using `Qdrant/bm25`, a traditional keyword-based retrieval model leveraging term frequency-inverse document frequency (TF-IDF) techniques.
- Late Interaction Embeddings – Based on `colbert-ir/colbertv2.0`, a deep-learning-based ranking model that refines search results through late interaction mechanisms.

#### Indexing Workflow
- Chunking – Data is split into structured segments using predefined chunk size.
- Embedding Generation – Each chunk is encoded using the three embedding models.
- Storage – The resulting vectors are stored in Qdrant, facilitating hybrid retrieval during inference.

## Inference

# Metrics and evaluation

# Future Improvements
