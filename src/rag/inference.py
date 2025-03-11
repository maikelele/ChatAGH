import os
from dotenv import load_dotenv

from rag.vector_stores.quadrant_vector_store import QuadrantCloudVectorStore
from src.rag.embeddings.google_embeddings import GoogleEmbeddings
from src.rag.retrievers.primitive_retriever import PrimitiveRetriever
from src.rag.models.google_model import GoogleModel

ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"

if __name__ == "__main__":
    load_dotenv(dotenv_path=ENV_PATH)

    query = "W jakiej sytuacji rozpocznie się procedura wydawania tabletek?"

    embeddings = GoogleEmbeddings()

    vector_store = QuadrantCloudVectorStore(
        api_key=os.getenv("QUADRANT_API_KEY"),
        collection_name=os.getenv("QUADRANT_COLLECTION_NAME"),
        url=os.getenv("QUADRANT_URL"),
        embedding_fn=embeddings
    )

    retriever = PrimitiveRetriever(vector_store=vector_store)
    source_docs = retriever.invoke(query)

    model = GoogleModel()
    res = model.generate_from_documents(query, source_docs)