import os

from google import genai
from langchain_core.documents import Document

from POC.rag.src.models.base_model import BaseModel
from POC.rag.src.utils.prompts import RAG_PROMPT_TEMPLATE


class GoogleModel(BaseModel):
    def __init__(self, model_name="gemini-2.0-flash-001"):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model_name

    def generate_from_documents(self, question: str, documents: list[Document]):
        contents = [
            RAG_PROMPT_TEMPLATE.format(
                SOURCE_DOCUMENTS=documents,
                USER_QUESTION=question
            )
        ]
        return self._inference(contents)

    def generate(self, question: str):
        raise NotImplementedError()

    def _inference(self, contents):
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        ).text

        return response