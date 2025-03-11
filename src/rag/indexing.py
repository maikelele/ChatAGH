import sys

from dotenv import load_dotenv
from src.rag.utils.utils import load_from_json
from src.rag.utils.watsonx_client import watsonx_client
from src.rag.chunkers.langchain_chunker import LangChainChunker
from src.rag.embeddings.watsonx_embeddings import WatsonXEmbeddings
from src.rag.embeddings.google_embeddings import GoogleEmbeddings
from src.rag.vector_stores.chroma_vector_store import ChromaVectorStore

ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
DATA_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/data/httpswww.agh.edu.pl.json"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"


if __name__ == "__main__":
    load_dotenv(dotenv_path=ENV_PATH)

    # client = watsonx_client()

    data = load_from_json(DATA_PATH, max_docs=50)

    chunker = LangChainChunker(500, 100)
    chunks = chunker.chunk(data)

    # embeddings = WatsonXEmbeddings(client)
    embeddings = GoogleEmbeddings()
    vectors = embeddings.embed_documents(chunks, batch_size=100)

    vector_store = ChromaVectorStore(collection_name="agh_edu", persist_directory=VECTOR_STORE_PATH)
    vector_store.add_documents(chunks, vectors)
