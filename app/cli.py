from asyncio import to_thread

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Select, TextArea

from .qa_engine import get_qa_chain
from .utils import get_available_models

MODELS_DIR = "./models"
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ID_LLM = "llm-path"
ID_EMB = "embedding-model"
ID_URL = "paper-url"
ID_QUESTION = "question"
ID_OUTPUT = "output"


async def answer_question(
    paper_url: str, emb_model: str, llm_path: str, question: str
) -> str:
    """
    Generates an answer to a question based on the content of a research paper.

    This function initializes a QA chain using the provided embedding model, language model path,
    and paper URL, then invokes the chain with the given question.

    Args:
        paper_url (str): URL of the research paper (either web URL or local path).
        emb_model (str): Name of the embedding model to use.
        llm_path (str): Path to the local language model to use.
        question (str): User inputted question.

    Returns:
        str: The answer generated by the QA chain.
    """
    qa_chain = await to_thread(get_qa_chain, paper_url, emb_model, llm_path)
    return await to_thread(qa_chain.invoke, question)


class PaperQueryCLI(App):
    """
    A Textual TUI application that allows users to input a paper URL and a question,
    and receive an answer using a selected language model and embedding model.
    """

    CSS_PATH = "cli.tcss"

    def compose(self) -> ComposeResult:
        """Compose and return the layout of the application."""
        yield Vertical(
            Select(get_available_models(MODELS_DIR), allow_blank=False, id=ID_LLM),
            Input(DEFAULT_EMB_MODEL, placeholder="Enter embedding model...", id=ID_EMB),
            Input(placeholder="Enter paper URL...", id=ID_URL),
            Input(placeholder="Ask your question here...", id=ID_QUESTION),
            TextArea(id=ID_OUTPUT, read_only=True),
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Handle user input when the 'question' field is submitted.

        Retrieves values from the input fields, invokes the QA chain, and updates the output area.
        """
        # Only proceed if the submitted input is the question field
        if event.input.id != ID_QUESTION:
            return

        # Extract inputs from UI components
        llm_path = self.query_exactly_one(f"#{ID_LLM}", Select).value
        emb_model = self.query_exactly_one(f"#{ID_EMB}", Input).value
        paper_url = self.query_exactly_one(f"#{ID_URL}", Input).value
        question = event.value

        output_widget = self.query_exactly_one(f"#{ID_OUTPUT}", TextArea)
        question_widget = self.query_exactly_one(f"#{ID_QUESTION}", Input)

        if not paper_url or not question:
            output_widget.text = "Please provide both a paper URL and a question."
            return

        output_widget.text = "Processing..."

        # Generate answer using the QA chain
        try:
            response = await answer_question(paper_url, emb_model, llm_path, question)
        except Exception as e:
            output_widget.text = f"Error: {e}"
            return

        # Display the response and clear the question input field
        output_widget.text = response
        question_widget.value = ""
