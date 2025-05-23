import contextlib
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from .components import SYSTEM_PROMPT_TEMPLATE, make_cached_getters
from .utils import format_chunks

# Create cached getter functions for the vector store and LLM
get_vectorstore, get_llm = make_cached_getters()


def get_qa_chain(url: str, emb_model: str, llm_path: str) -> Runnable:
    """
    Constructs a question-answering (QA) chain for processing user queries on a given paper.

    This function sets up a LangChain pipeline that:
    - Retrieves relevant document chunks from a vector store.
    - Formats the retrieved chunks.
    - Combines them with the user's question into a prompt.
    - Sends the prompt to a language model.
    - Parses and returns the model's string output.

    Args:
        url (str): The URL or local path of the paper to process.
        emb_model (str): The embedding model name to use for building the vector store.
        llm_path (str): The path to the language model used to answer questions.

    Returns:
        Runnable: A LangChain Runnable object representing the full QA chain.
    """
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

    # Suppress stderr temporarily to avoid Textual app issues (i.e. warning noise)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        vectorstore = get_vectorstore(url, emb_model)
        llm = get_llm(llm_path)

    qa_chain = (
        {
            "question": RunnablePassthrough(),
            "chunks": vectorstore.as_retriever() | format_chunks,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain
