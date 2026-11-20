"""
llm.py - Small helpers for talking to the local Ollama model.

What this file does, in simple words:
- build_chat_model() creates the Ollama client. The planner, workers,
  reviewer and writer all use it, so their settings stay in one place.
- ask_for_structured() asks the model for a JSON answer of a fixed shape,
  checks it, and asks again (saying what was wrong) if it does not fit.
  Small local models sometimes write broken JSON, so this matters.
- connection_hint() adds a helpful tip when Ollama cannot be reached.
"""

from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from app import config

Shape = TypeVar("Shape", bound=BaseModel)


class InvalidAnswerError(Exception):
    """Raised when the model gave no valid answer after every try."""


def build_chat_model(temperature: float, max_tokens: int, timeout_seconds: int) -> Any:
    """Create an Ollama chat client with our settings.

    Building the client does not contact Ollama, so tests can call this
    even when Ollama is not running.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=config.OLLAMA_NUM_CTX,  # every caller uses the same size, so
        # Ollama never has to reload the model to change it
        num_predict=max_tokens,
        client_kwargs={"timeout": timeout_seconds},
    )


def _read_answer(answer: Any, shape: type[Shape]) -> Shape:
    """Check the answer has the right shape. Raises ValueError if not."""
    if isinstance(answer, shape):
        return answer
    if isinstance(answer, dict):
        return shape.model_validate(answer)  # raises ValidationError (a ValueError)
    raise ValueError(f"expected a JSON object, got {type(answer).__name__}")


def ask_for_structured(
    llm: Any,
    messages: list[tuple[str, str]],
    shape: type[Shape],
    attempts: int,
) -> Shape:
    """Ask llm for an answer of this shape, retrying when it does not fit.

    llm:      a model made with .with_structured_output(shape), or a test fake.
    messages: the system and human messages to send.
    attempts: how many tries in total.

    Raises InvalidAnswerError if every try gave a bad answer. Any other
    error (Ollama not running, timeout...) is raised at once, unchanged,
    because asking again would not help.
    """
    last_problem = ""
    current = list(messages)

    for _attempt in range(attempts):
        try:
            return _read_answer(llm.invoke(current), shape)
        except (OutputParserException, ValueError) as error:
            # A bad answer (broken JSON, missing field, number out of range).
            # Tell the model what was wrong and ask again.
            last_problem = str(error)[:500]
            current = list(messages) + [
                (
                    "human",
                    f"Your last answer was not valid: {last_problem}\n"
                    "Reply again with ONLY the JSON object, in the format shown.",
                )
            ]

    raise InvalidAnswerError(
        f"no valid answer after {attempts} tries. Last problem: {last_problem}"
    )


def connection_hint(error: Exception) -> str:
    """Extra help when the problem is that Ollama cannot be reached."""
    if isinstance(error, ConnectionError) or "connect" in str(error).lower():
        return (
            f" Is Ollama running at {config.OLLAMA_BASE_URL}? "
            "Start it with:  ollama serve"
        )
    return ""
