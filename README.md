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
<img width="971" alt="Screenshot 2025-03-16 at 17 05 31" src="https://github.com/user-attachments/assets/fc19e15d-c8ac-48e8-8ab8-f17e2bc16d16" />


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
Short queries often lack sufficient detail to retrieve the most relevant chunks. To address this, each query is expanded using a properly prompted LLM. This process ensures that the question covers more information, improving retrieval effectiveness.

### Hybrid Search Retrieval
As mentioned earlier, chunks are stored with three types of embedding vectors (`dense`, `sparse`, and `late interaction`). Our hybrid search implementation includes:

- **Prefetching**: Uses dense and sparse embeddings to extract the most similar chunks to the augmented query.
- **Late Interaction Filtering**: The `late_interaction` model compares the query vector against vectors retrieved in the prefetch step. A specified number of the most relevant chunks is then returned.

### Multi-Collection Search
To optimize retrieval performance and quality, data is segmented into thematically similar groups, each stored in a separate vector store collection. During retrieval, our system queries all collections separately using hybrid search, retrieving a larger set of chunks. Next, we compare the dense embedding vectors of the retrieved chunks against the augmented query vector and return the most similar ones. This approach ensures that the most relevant chunks are retrieved across the entire dataset while maintaining high performance.

### Enhanced Retrieval Generation
Since retrieval may not always yield all necessary information to answer a query, we use an additional LLM during the generation step. This model assesses whether more data is required. If so, it generates additional questions, and the vector store is queried again to refine the retrieved sources. This iterative process continues until either all necessary information is found or a predefined search limit is reached.

Finally, following the classical RAG approach, we use a properly prompted LLM to generate the final answer based on the retrieved information.

# Metrics and evaluation

# Future Improvements
