"""
reviewer.py - Scores each worker's output from 1 to 5. (Phase 4)

What this file does, in simple words:
- After the workers finish, the reviewer reads each worker's output next to
  the sub-task that worker was given.
- It asks the model for a score (1 = useless, 5 = great) and a short reason.
- The writer then trusts high-scored outputs more than low-scored ones.

Safety:
- A worker that failed is not sent to the model. It simply gets no score.
- A review that goes wrong (bad JSON, Ollama down) never stops the run.
  That worker just gets no score and a note saying why.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from app import config
from app.llm import InvalidAnswerError, ask_for_structured, build_chat_model, connection_hint
from app.workers import WorkerResult

# The most characters of one worker's output the reviewer reads.
_MAX_OUTPUT_CHARS = 8000


# --- The shape of a review --------------------------------------------------
class Review(BaseModel):
    """What the model must answer. Feedback comes first, so the model
    thinks about the output before it picks a number."""

    feedback: str = Field(
        description="One to three sentences: what is good and what is missing or wrong."
    )
    score: int = Field(
        ge=config.REVIEW_MIN_SCORE,
        le=config.REVIEW_MAX_SCORE,
        description=f"A whole number from {config.REVIEW_MIN_SCORE} to {config.REVIEW_MAX_SCORE}.",
    )


class WorkerReview(BaseModel):
    """The review we store for one worker."""

    role: str
    score: int | None = None  # None = not reviewed (the worker or review failed)
    feedback: str = ""

    @property
    def reviewed(self) -> bool:
        return self.score is not None


# --- The instructions we give the model -------------------------------------
REVIEWER_SYSTEM_PROMPT = """\
You are the reviewer in a multi-agent system. A worker agent was given one \
sub-task. You read its output and score how well it did that sub-task.

Scoring scale:
5 = does the whole sub-task. Specific, clear, and gives sources for facts \
taken from the web.
4 = good, with small gaps.
3 = does part of the sub-task, or is vague, or has important gaps.
2 = mostly off-topic, very thin, or probably wrong in places.
1 = does not do the sub-task at all.

Judge only the output against its sub-task. You cannot check the web \
yourself, so score lower for claims that look made up or have no source.

Answer format: reply with ONE JSON object and nothing else, like this:
{"feedback": "Covers all seven days with clear steps, but gives no sources.", "score": 4}
"""


# --- Talking to the model ---------------------------------------------------
def build_reviewer_llm() -> Any:
    """Create the Ollama client that returns a Review object."""
    llm = build_chat_model(
        temperature=0,  # the same output gets the same score
        max_tokens=config.REVIEWER_MAX_TOKENS,
        timeout_seconds=config.REVIEWER_TIMEOUT_SECONDS,
    )
    return llm.with_structured_output(Review, method="json_schema")


def review_one(result: WorkerResult, llm: Any) -> WorkerReview:
    """Review one worker's output. Never raises an error."""
    if not result.succeeded:
        return WorkerReview(role=result.role, feedback=f"Not reviewed: the worker failed ({result.error})")

    messages = [
        ("system", REVIEWER_SYSTEM_PROMPT),
        (
            "human",
            f"Sub-task given to the worker:\n{result.task}\n\n"
            f"The worker's output:\n{result.output[:_MAX_OUTPUT_CHARS]}",
        ),
    ]
    try:
        review = ask_for_structured(llm, messages, Review, config.REVIEWER_MAX_ATTEMPTS)
    except InvalidAnswerError as error:
        return WorkerReview(role=result.role, feedback=f"Not reviewed: the reviewer gave {error}")
    except Exception as error:  # noqa: BLE001 - Ollama down, timeout...
        return WorkerReview(
            role=result.role,
            feedback=f"Not reviewed: {type(error).__name__}: {error}.{connection_hint(error)}",
        )
    return WorkerReview(role=result.role, score=review.score, feedback=review.feedback.strip())


# --- The one function other files call --------------------------------------
def review_all(results: list[WorkerResult], llm: Any = None) -> list[WorkerReview]:
    """Review every worker's output. Returns one review per result, in the same order.

    llm: leave empty in normal use. Tests pass a fake model here.
    """
    if not any(result.succeeded for result in results):
        # Nothing to score, so do not even start the model.
        return [review_one(result, llm) for result in results]

    if llm is None:
        llm = build_reviewer_llm()

    # Review all outputs at the same time. (If Ollama can only do one at a
    # time, it simply queues them.) map() keeps the original order.
    with ThreadPoolExecutor(max_workers=len(results)) as pool:
        return list(pool.map(lambda result: review_one(result, llm), results))
