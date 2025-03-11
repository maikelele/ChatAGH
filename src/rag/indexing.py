import os

from dotenv import load_dotenv
from src.rag.utils.utils import load_from_json
from src.rag.chunkers.langchain_chunker import LangChainChunker
from src.rag.embeddings.google_embeddings import GoogleEmbeddings
from src.rag.vector_stores.quadrant_vector_store import QuadrantCloudVectorStore

ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
DATA_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/data/httpswww.agh.edu.pl.json"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"


if __name__ == "__main__":
    load_dotenv(dotenv_path=ENV_PATH)
    data = load_from_json(DATA_PATH)

    chunker = LangChainChunker(1000, 100)
    chunks = chunker.chunk(data)

    embeddings = GoogleEmbeddings()
    vectors = embeddings.embed_documents(chunks, batch_size=100)

    vector_store = QuadrantCloudVectorStore(
        api_key=os.getenv("QUADRANT_API_KEY"),
        collection_name=os.getenv("QUADRANT_COLLECTION_NAME"),
        url=os.getenv("QUADRANT_URL"),
        embedding_fn=embeddings,
        upload_batch_size=1000
    )
    vector_store.add_documents(chunks, vectors)

