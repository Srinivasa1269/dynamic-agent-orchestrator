"""
test_writer.py - Checks the writer with fake models (no Ollama needed).
"""

from langchain_core.messages import AIMessage

from app import config
from app.reviewer import WorkerReview
from app.workers import WorkerResult
from app.writer import build_writer_input, build_writer_llm, write_report


class FakeWriterLLM:
    """Returns one fixed answer, or raises it if it is an Exception."""

    def __init__(self, answer):
        self.answer = answer
        self.received = None

    def invoke(self, messages):
        self.received = messages
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


RESULTS = [
    WorkerResult(role="prices", task="Find car prices.", output="EVs cost more. (a.com)"),
    WorkerResult(role="emissions", task="Find car emissions.", error="Timed out"),
]
REVIEWS = [
    WorkerReview(role="prices", score=4, feedback="Good."),
    WorkerReview(role="emissions", feedback="Not reviewed: the worker failed"),
]


def test_report_is_the_models_text() -> None:
    report = write_report("Compare cars", RESULTS, REVIEWS, llm=FakeWriterLLM(AIMessage(content=" # Report ")))
    assert report.text == "# Report"
    assert not report.fallback and report.error == ""


def test_writer_input_has_task_outputs_scores_and_failed_parts() -> None:
    text = build_writer_input("Compare cars", RESULTS, REVIEWS)
    assert "Compare cars" in text
    assert "EVs cost more. (a.com)" in text
    assert "score: 4/5" in text
    assert "FAILED" in text and "Find car emissions." in text


def test_long_outputs_are_cut_to_fit() -> None:
    huge = WorkerResult(role="big", task="t", output="x" * (config.WRITER_MAX_INPUT_CHARS * 3))
    text = build_writer_input("Task", [huge], [])
    assert len(text) < config.WRITER_MAX_INPUT_CHARS + 1000
    assert "cut to fit" in text


def test_writer_failure_falls_back_to_the_raw_outputs() -> None:
    report = write_report("Compare cars", RESULTS, REVIEWS, llm=FakeWriterLLM(TimeoutError("too slow")))
    assert report.fallback
    assert "too slow" in report.error
    assert "EVs cost more." in report.text
    assert "This part failed: Timed out" in report.text


def test_empty_answer_falls_back_too() -> None:
    report = write_report("Compare cars", RESULTS, REVIEWS, llm=FakeWriterLLM(AIMessage(content="  ")))
    assert report.fallback


def test_every_worker_failed_needs_no_model() -> None:
    # llm=None would normally build a real client; here no model is needed.
    report = write_report("Compare cars", [RESULTS[1]], [REVIEWS[1]])
    assert report.fallback
    assert "every worker failed" in report.text


def test_real_writer_client_can_be_built_without_calling_ollama() -> None:
    assert build_writer_llm() is not None
