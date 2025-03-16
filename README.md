# Contents
- [Overview](#overview)
- [Developer guide](#developer-guide)
- [Data processing](#data-processing)
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
Add `.env` file with your credentials in src directory, you can find required credentials in `.env.template`

### Run streamlit app
```
streamlit run src/streamlit_app.py
```

# Data processing
## Data sources 
## Scraping
## Processing

# RAG implementation

## Indexing

### Chunking

Once the data has been preprocessed and segmented, the next step involves breaking it into smaller, more manageable chunks. This process, known as chunking, is essential for efficient indexing and retrieval in large-scale information systems. To achieve this, the LangChain Recursive Character Text Splitter is utilized, spliting processed data using specified chunking parameters. 

### Vector Store

After chunking, the generated chunks are stored in a dedicated vector database for efficient retrieval. In this implementation, Qdrant is employed as the vector store due to its high performance and optimization for hybrid search scenarios. Qdrant provides fast and scalable storage of vector embeddings, enabling both semantic and keyword-based search functionalities.

Each segment chunks are stored in a distinct collection within the vector store. This approach is necessary because the dataset is too large to be stored in a single collection without compromising performance and retrieval quality. Additionally, separating the data into multiple collections allows for thematic categorization, enabling more precise and context-aware search results.

To enhance search effectiveness, a hybrid search retrieval mechanism is implemented. This approach leverages multiple types of embeddings for each chunk, ensuring that the system can handle diverse query types and return the most relevant results. The embeddings used include:

- **Dense Embeddings** – Derived from `distiluse-base-multilingual-cased-v1`, a transformer-based model designed to generate sentence embeddings optimized for capturing semantic similarity.
- **Sparse Embeddings** – Created using `Qdrant/bm25`, a traditional keyword-based retrieval model that utilizes Term Frequency-Inverse Document Frequency (TF-IDF) techniques to improve lexical matching.
- **Late Interaction Embeddings** – Based on `colbert-ir/colbertv2.0`, a deep learning model that refines search results by considering the interaction between query and document representations at a later stage in the ranking process.

By combining these embedding techniques, the retrieval system is capable of delivering high-quality search results that balance both semantic relevance and lexical precision.





## Inference

### Query Augmentation
Short queries sometimes can carry too little information to retrieve most relevent chunks, to levarage this each query is expaned. We are using accordingly prompted LLM to rewrite question to cover more information.

### Hybrid Search Retrieval
As mentioned above, chunks are stored wtih three embedding vector (`dense`, `spare` and `late interaction`). Our hybrid search implementation includes:
- Prefetch: uses dense and sparse embeddings, extracts most similar to the augmented query.
- We are using `late_interaction` model to compare query vector agains vectors extracted from fetch step, specified number of most similar ones is returned. 

### Multi Collection Search


### Enhanced Retrieval Generation


# Metrics and evaluation

# Future Improvements
