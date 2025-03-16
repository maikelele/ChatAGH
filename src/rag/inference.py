from dotenv import load_dotenv
from langchain_core.documents import Document

from rag.vector_store.multi_collection_search import MultiCollectionSearch
from rag.models.google_genai_models import (
    QueryAugmentationModel,
    EnhanceSearchModel,
    AnswerGenerationModel
)

ENV_PATH = ".env"
NUM_RETRIEVED_CHUNKS = 20
MAX_SEARCH_ITERATIONS = 5


def inference(query):
    load_dotenv(dotenv_path=ENV_PATH)

    query_augmentation_model = QueryAugmentationModel()
    augmented_query = query_augmentation_model.generate(query)

    vector_store = MultiCollectionSearch()
    source_docs = vector_store.search(query, NUM_RETRIEVED_CHUNKS)

    enhance_search_model = EnhanceSearchModel()
    summaries = []
    for i in range(MAX_SEARCH_ITERATIONS):
        summary, questions = enhance_search_model.generate(augmented_query, context=source_docs)

        if not (summary and questions):
            break

        summaries.append({"text": summary})

        questions = " \n".join(questions)
        source_docs = vector_store.search(questions, k=NUM_RETRIEVED_CHUNKS)

    source_docs.extend(summaries)

    answer_generation_model = AnswerGenerationModel()
    final_response = answer_generation_model.generate(augmented_query, context=source_docs)

    return final_response, source_docs


if __name__ == "__main__":
    query = "Czy mogę brać udział w rekrutacji będąc niepełnoletnim?"
    print(inference(query))
