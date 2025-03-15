from typing import List

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.retrievers.base_retriever import BaseRetriever

DOCUMENTS_DESCRIPTION = """
Document provides detailed information about study programs,
 learning outcomes, and course (module) descriptions available at the AGH University of Science and Technology.
"""

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="title",
        description="Title of the document",
        type="string",
    ),
    AttributeInfo(
        name="url",
        description="URL of the source of the document",
        type="string"
    ),
    AttributeInfo(
        name="description",
        description="Description of the document",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="Date when the source of the document was published",
        type="string",
    ),
    AttributeInfo(
        name="tags",
        description="Tags associated with the document",
        type="list",
    )
]


class SQRetriever(BaseRetriever):
    def __init__(self, vector_store):
        self.retriever = SelfQueryRetriever.from_llm(
            ChatGoogleGenerativeAI(model="gemini-2.0-flash-001"),
            vector_store.get_langchain_vector_store(),
            DOCUMENTS_DESCRIPTION,
            METADATA_FIELD_INFO
        )

    def invoke(self, query) -> List[Document]:
        return self.retriever.invoke(query)