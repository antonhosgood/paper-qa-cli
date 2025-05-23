from typing import Callable, Tuple

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

SYSTEM_PROMPT_TEMPLATE = """
You are a scientific research assistant.

You help users understand scientific papers by answering their questions using only the content of the paper.
With each question, you receive a few relevant chunks from the research paper.
Each chunk begins with 'Chunk {{i}}:' (where i is the chunk number) followed by some relevant information from the paper.

When a user asks a question, provide a clear, concise answer based on the provided chunks.
If the answer is not in the paper, say so clearly. Do not speculate.

Question:
{question}

Chunks:
{chunks}

Answer:
"""


def create_vectorstore(url: str, emb_model_name: str) -> FAISS:
    """
    Create a FAISS vector store from a PDF document loaded from a URL or file path.

    Args:
        url (str): The URL or local file path of the PDF document.
        emb_model_name (str): The embedding model name used for chunking and embeddings.

    Returns:
        FAISS: A vector store instance populated with document chunks and their embeddings.
    """
    loader = PyPDFLoader(url)
    text_splitter = SentenceTransformersTokenTextSplitter(model_name=emb_model_name)

    chunks = loader.load_and_split(text_splitter)
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)
    return FAISS.from_documents(chunks, embeddings)


def create_llm(model_path: str) -> LlamaCpp:
    """
    Create and configure a LlamaCpp language model instance.

    Args:
        model_path (str): Path to the local model file.

    Returns:
        LlamaCpp: A configured LlamaCpp model instance.
    """
    return LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        temperature=0.7,
        top_p=0.95,
        verbose=False,
    )


def make_cached_getters() -> Tuple[
    Callable[[str, str], FAISS], Callable[[str], LlamaCpp]
]:
    """Returns cached getter functions for the vectorstore and language model to avoid reloading them multiple times."""
    cache = {
        "vectorstore": {},
        "llm": None,
        "llm_path": None,
    }

    # Different vector stores will be cached together
    def get_vectorstore(url: str, emb_model_name: str) -> FAISS:
        key = (url, emb_model_name)
        if key not in cache["vectorstore"]:
            cache["vectorstore"][key] = create_vectorstore(url, emb_model_name)
        return cache["vectorstore"][key]

    # If a different LLM is loaded, the cache will be cleared
    def get_llm(llm_path: str) -> LlamaCpp:
        if llm_path != cache["llm_path"]:
            cache["llm"] = create_llm(llm_path)
            cache["llm_path"] = llm_path
        return cache["llm"]

    return get_vectorstore, get_llm
