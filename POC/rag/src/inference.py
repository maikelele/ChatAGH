from dotenv import load_dotenv
from POC.rag.src.utils.watsonx_client import watsonx_client
from POC.rag.src.embeddings.watsonx_embeddings import WatsonXEmbeddings
from POC.rag.src.embeddings.google_embeddings import GoogleEmbeddings
from POC.rag.src.retrievers.primitive_retriever import PrimitiveRetriever
from POC.rag.src.vector_stores.chroma_vector_store import ChromaVectorStore
from POC.rag.src.models.watsonx_model import WatsonXModel
from POC.rag.src.models.google_model import GoogleModel

ENV_PATH = "/Users/wnowogorski/PycharmProjects/ChatAGH/POC/config/.env"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/ChatAGH/POC"

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