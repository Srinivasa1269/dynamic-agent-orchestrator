"""
writer.py - Merges all worker outputs into one final report. (Phase 4)

What this file does, in simple words:
- It gives the model the user's task, every worker's output, and the
  reviewer's score for each one.
- The model writes ONE clear answer for the user. It trusts high-scored
  outputs more and keeps the sources the workers found.

Safety: the user always gets a report.
- If every worker failed, we say so (no model call needed).
- If the writer itself fails (Ollama down, timeout, empty answer), we fall
  back to showing the workers' own outputs one after another.
"""

from typing import Any

from pydantic import BaseModel

from app import config
from app.llm import build_chat_model, connection_hint
from app.reviewer import WorkerReview
from app.workers import WorkerResult


class Report(BaseModel):
    """The final answer for the user."""

    text: str  # the report itself, in Markdown
    fallback: bool = False  # True = the writer failed; text is the raw outputs
    error: str = ""  # why the writer failed (empty if it worked)


# --- The instructions we give the model -------------------------------------
WRITER_SYSTEM_PROMPT = f"""\
You are the writer in a multi-agent system. Several worker agents each did \
one part of the user's task. A reviewer scored each output from \
{config.REVIEW_MIN_SCORE} (bad) to {config.REVIEW_MAX_SCORE} (great). Merge \
their outputs into ONE final answer for the user.

Rules:
- Answer the user's task directly. Fit the format to the task: a plan, a \
comparison, a short answer, and so on.
- Use only the facts in the worker outputs. Do not add new facts from memory.
- Trust higher-scored outputs more. Use a fact from an output scored 1 or 2 \
only if another output supports it.
- Remove repeats. If outputs disagree, say so briefly.
- Keep the source URLs the workers gave next to their facts, and list them \
at the end under a "Sources" heading.
- If part of the task is missing because a worker failed or found nothing, \
say so in one line.
- Write in clear, simple English with Markdown headings and bullet points.
- Do not mention the workers, the reviewer or the scores.
"""


def build_writer_llm() -> Any:
    """Create the Ollama client the writer uses."""
    return build_chat_model(
        temperature=0.3,  # a little freedom for natural writing
        max_tokens=config.WRITER_MAX_TOKENS,
        timeout_seconds=config.WRITER_TIMEOUT_SECONDS,
    )


# --- Preparing what the writer reads ----------------------------------------
def _trim(text: str, limit: int) -> str:
    """Cut text to at most `limit` characters, marking the cut."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n[... cut to fit ...]"


def build_writer_input(
    task: str, results: list[WorkerResult], reviews: list[WorkerReview]
) -> str:
    """Write the message the writer model reads: the task plus every output."""
    scores = {review.role: review for review in reviews}
    succeeded = [result for result in results if result.succeeded]
    failed = [result for result in results if not result.succeeded]

    # Share the space fairly, so one long output cannot push out the others.
    per_output = config.WRITER_MAX_INPUT_CHARS // max(len(succeeded), 1)

    parts = [f"The user's task:\n{task}\n"]
    for result in succeeded:
        review = scores.get(result.role)
        score = f"{review.score}/{config.REVIEW_MAX_SCORE}" if review and review.reviewed else "not scored"
        parts.append(
            f"--- Output from {result.role} (score: {score}) ---\n"
            f"Its part of the task: {result.task}\n\n"
            f"{_trim(result.output, per_output)}\n"
        )
    if failed:
        missing = "\n".join(f"- {result.task}" for result in failed)
        parts.append(f"--- These parts FAILED and have no output ---\n{missing}\n")
    parts.append("Now write the final answer for the user.")
    return "\n".join(parts)


def _fallback_report(results: list[WorkerResult], error: str) -> Report:
    """When the writer fails: show each worker's own output instead."""
    sections = [
        "*The writer could not merge the results, so here is each part on its own.*"
    ]
    for result in results:
        body = result.output if result.succeeded else f"This part failed: {result.error}"
        sections.append(f"## {result.role}\n\n{body}")
    return Report(text="\n\n".join(sections), fallback=True, error=error)


# --- The one function other files call --------------------------------------
def write_report(
    task: str,
    results: list[WorkerResult],
    reviews: list[WorkerReview],
    llm: Any = None,
) -> Report:
    """Merge every worker's output into one final report. Never raises an error.

    llm: leave empty in normal use. Tests pass a fake model here.
    """
    if not any(result.succeeded for result in results):
        lines = "\n".join(f"- {result.role}: {result.error}" for result in results)
        return Report(
            text=f"No report could be written, because every worker failed.\n\n{lines}",
            fallback=True,
            error="every worker failed",
        )

    if llm is None:
        llm = build_writer_llm()

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", build_writer_input(task, results, reviews)),
    ]
    try:
        answer = llm.invoke(messages)
    except Exception as error:  # noqa: BLE001 - Ollama down, timeout...
        return _fallback_report(
            results, f"{type(error).__name__}: {error}.{connection_hint(error)}"
        )

    # .text gives only the written answer, leaving out any "thinking" parts.
    text = str(getattr(answer, "text", "")).strip()
    if not text:
        return _fallback_report(results, "the writer returned no text")
    return Report(text=text)
