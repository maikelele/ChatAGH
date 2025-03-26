import os
from multiprocessing import Pool

from dotenv import load_dotenv
from rag.utils.utils import load_data
from rag.chunkers.langchain_chunker import LangChainChunker
from rag.vector_store.qdrant_hybrid_search import QdrantHybridSearchVectorStore
from rag.vector_store.pinecone_hybrid_search import PineconeHybridSearchVectorStore

ENV_PATH = ".env"
DATA_PATH = ""


def indexing_single(data_path, collection_name, chunk_size=1000, chunk_overlap=100, max_chunks=None):
    """
    Index documents from a single data path into a specific vector store collection

    Args:
        data_path (str): Path to the data file
        collection_name (str): Name of the collection in vector store
        chunk_size (int): Size of chunks for document splitting
        chunk_overlap (int): Overlap between chunks

    Returns:
        tuple: (collection_name, number of chunks)
    """
    load_dotenv(dotenv_path=ENV_PATH)
    data = load_data(data_path)

    chunker = LangChainChunker(chunk_size, chunk_overlap, remove_duplicates=True)
    chunks = chunker.chunk(data)

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"Generated {len(chunks)} chunks from {data_path}")

    vector_store = PineconeHybridSearchVectorStore(os.environ["PINECONE_API_KEY"], "chatagh")

    # vector_store = QdrantHybridSearchVectorStore(collection_name=collection_name)
    vector_store.indexing(chunks)

    print(f"Indexed to collection: {collection_name}")
    return (collection_name, len(chunks))


def parallel_indexing(data_paths, collection_names, chunk_size=1000, chunk_overlap=100, num_processes=None):
    """
    Index documents from multiple data paths to specified collections in parallel

    Args:
        data_paths (list): List of paths to data files
        collection_names (list): List of collection names (must match length of data_paths)
        chunk_size (int): Size of chunks for document splitting
        chunk_overlap (int): Overlap between chunks
        num_processes (int, optional): Number of parallel processes. Defaults to CPU count.

    Returns:
        list: List of tuples (collection_name, chunk_count) for each data path
    """
    if len(data_paths) != len(collection_names):
        raise ValueError("The number of data paths must match the number of collection names")

    process_args = [(data_paths[i], collection_names[i], chunk_size, chunk_overlap)
                    for i in range(len(data_paths))]

    if num_processes is None:
        num_processes = os.cpu_count()

    num_processes = min(num_processes, len(data_paths))

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(indexing_single, process_args)

    total_chunks = sum(count for _, count in results)
    print(f"Total chunks indexed across {len(results)} collections: {total_chunks}")

    return results


if __name__ == "__main__":
    data_paths = [
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/miasteczko_akademik_agh",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/agh_edu_1",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/agh_edu_2",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/agh_edu_3",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/agh_edu_4",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/rekrutacja_sylabusy_agh",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/historia_agh",
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/dss_agh"
    ]

    collection_names = [
        "miasteczko_akademik_agh",
        "agh_edu_1",
        "agh_edu_2",
        "agh_edu_3",
        "agh_edu_4",
        "rekrutacja_sylabusy_agh",
        "historia_agh",
        "dss_agh"
    ]

    # results = parallel_indexing(data_paths, collection_names, num_processes=1)

    i = 5
    result = indexing_single(
        "/Users/wnowogorski/PycharmProjects/CHAT_AGH/src/collections/rekrutacja_sylabusy_agh",
        "chatagh",
        chunk_size=1500,
        chunk_overlap=0
    )

    print("\nIndexing Summary:")
    for collection_name, count in result:
        print(f"Collection '{collection_name}': {count} chunks")
