import sys

from dotenv import load_dotenv
from rag.src.utils.utils import load_from_json
from rag.src.utils.watsonx_client import watsonx_client
from rag.src.chunkers.langchain_chunker import LangChainChunker
from rag.src.embeddings.watsonx_embeddings import WatsonXEmbeddings
from rag.src.embeddings.google_embeddings import GoogleEmbeddings
from rag.src.vector_stores.chroma_vector_store import ChromaVectorStore

ENV_PATH = "/Users/wnowogorski/PycharmProjects/ChatAGH/POC/config/.env"
DATA_PATH = "/Users/wnowogorski/PycharmProjects/ChatAGH/POC/data/httpswww.agh.edu.pl.json"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/ChatAGH/POC"


if __name__ == "__main__":
    load_dotenv(dotenv_path=ENV_PATH)

    # client = watsonx_client()

    data = load_from_json(DATA_PATH, max_docs=50)
    print(data)

    print(len(data))

    chunker = LangChainChunker(500, 100)
    chunks = chunker.chunk(data)
    print(len(chunks))

    # embeddings = WatsonXEmbeddings(client)
    embeddings = GoogleEmbeddings()
    vectors = embeddings.embed_documents(chunks, batch_size=100)
    print(len(vectors))

    vector_store = ChromaVectorStore(collection_name="agh_edu", persist_directory=VECTOR_STORE_PATH)
    vector_store.add_documents(chunks, vectors)
