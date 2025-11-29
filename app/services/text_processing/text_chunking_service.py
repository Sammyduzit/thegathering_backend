from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunkingService:
    """Service for chunking text using RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text chunking service.

        :param chunk_size: Maximum size of each chunk (~125 tokens)
        :param chunk_overlap: Overlap between chunks for context preservation
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        :param text: Text to chunk
        :return: List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = self.splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
