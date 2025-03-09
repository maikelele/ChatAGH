from typing import List, Dict, Any, Optional, Callable

from POC.rag.src.chunkers.base_chunker import BaseChunker

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainChunker(BaseChunker):
    """
    Chunker implementation for LangChain documents.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        length_function: Callable[[str], int] = len,
        add_start_index: bool = True
    ):
        """
        Initialize the LangChain document chunkers.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: String separators to split text on when possible
            length_function: Function to measure text length (default: character count)
            add_start_index: Whether to add a 'start_index' field to chunk metadata
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.length_function = length_function
        self.add_start_index = add_start_index

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""] + self.separators if self.separators else [],
            length_function=length_function
        )

    def chunk(self, documents: List[Any]) -> List[Any]:
        """
        Split LangChain documents into chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of LangChain Document objects, chunked
        """
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All documents must be LangChain Document objects")

        chunked_docs = self.text_splitter.split_documents(documents)

        if self.add_start_index:
            for doc in chunked_docs:
                if not doc.metadata:
                    doc.metadata = {}

                parent_id = doc.metadata.get("parent_id")
                parent_text = None
                if parent_id:
                    for original_doc in documents:
                        if original_doc.metadata.get("id") == parent_id:
                            parent_text = original_doc.page_content
                            break

                if parent_text and doc.page_content in parent_text:
                    doc.metadata["start_index"] = parent_text.index(doc.page_content)
                else:
                    doc.metadata["start_index"] = 0

                doc.metadata["chunk_index"] = chunked_docs.index(doc)

        return chunked_docs

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Split text string into LangChain Document chunks.

        Args:
            text: Text string to be chunked
            metadata: Optional metadata to include with each chunk

        Returns:
            List of LangChain Document objects
        """
        if metadata is None:
            metadata = {}

        document = Document(page_content=text, metadata=metadata)

        return self.chunk([document])
