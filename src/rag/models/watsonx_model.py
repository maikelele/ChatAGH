from typing import List

from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_core.documents import Document

from src.rag.models.base_model import BaseModel
from src.rag.utils.prompts import RAG_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME = "mistralai/mistral-large"

class WatsonXModel(BaseModel):
    def __init__(self, client, model_name=DEFAULT_MODEL_NAME):
        self.model = ModelInference(
            api_client=client,
            model_id=model_name,
            params={
                "max_new_tokens": 2000,
                "min_new_tokens": 1,
            },
        )

    def generate_from_documents(self, question: str, documents: List[Document]):
        documents_content = "\n\n".join([doc.page_content for doc in documents])
        prompt = RAG_PROMPT_TEMPLATE.format(
            SOURCE_DOCUMENTS=documents_content,
            USER_QUESTION=question,
        )
        generated_response = self.model.generate(prompt=prompt)

        return generated_response

    def generate(self, question: str):
        raise NotImplemented
