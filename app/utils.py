from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document


def get_available_models(path: str) -> List[Tuple[str, str]]:
    """
    Scans a directory for `.gguf` model files and returns their names and absolute paths.

    Args:
        path (str): Path to the directory containing model files.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the model filename and its corresponding
                               absolute path.
    """
    models_dir = Path(path).resolve()
    return sorted([(f.name, str(f)) for f in models_dir.glob("*.gguf")])


def format_chunks(chunks: List[Document]) -> str:
    """
    Formats a list of LangChain ``Document`` chunks into a numbered string format.

    Each chunk is prefixed with "Chunk {i}:" followed by the chunk's content.

    Args:
        chunks (List[Document]): A list of ``Document`` objects to format.

    Returns:
        str: A formatted string representation of the chunks.
    """
    return "\n".join(
        f"Chunk {i + 1}: {chunk.page_content}" for i, chunk in enumerate(chunks)
    )
