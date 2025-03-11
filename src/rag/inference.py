from dotenv import load_dotenv
from src.rag.utils.watsonx_client import watsonx_client
from src.rag.embeddings.watsonx_embeddings import WatsonXEmbeddings
from src.rag.embeddings.google_embeddings import GoogleEmbeddings
from src.rag.retrievers.primitive_retriever import PrimitiveRetriever
from src.rag.vector_stores.chroma_vector_store import ChromaVectorStore
from src.rag.models.watsonx_model import WatsonXModel
from src.rag.models.google_model import GoogleModel

ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"

if __name__ == "__main__":
    load_dotenv(dotenv_path=ENV_PATH)

    query = "W jakiej sytuacji rozpocznie się procedura wydawania tabletek?"

    # client = watsonx_client()
    embeddings = GoogleEmbeddings()

    vector_store = ChromaVectorStore(
        collection_name="agh_edu",
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    # embeddings = WatsonXEmbeddings(client)

    retriever = PrimitiveRetriever(vector_store=vector_store)
    source_docs = retriever.invoke(query)

    print(source_docs)

    model = GoogleModel()
    # model = WatsonXModel(client)
    res = model.generate_from_documents(query, source_docs)
    print(res)