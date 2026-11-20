"""
test_reviewer.py - Checks the reviewer with fake models (no Ollama needed).

The last test is marked "live" and uses the real local model.
"""

import pytest
from langchain_core.exceptions import OutputParserException

from app import config
from app.reviewer import Review, build_reviewer_llm, review_all, review_one
from app.workers import WorkerResult


class FakeReviewerLLM:
    """Returns the given answers one after another (the last one repeats)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0
        self.received = None

    def invoke(self, messages):
        self.received = messages
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return answer


GOOD = WorkerResult(role="researcher", task="Find Everest's height.", output="8,849 m (source: x.com)")
FAILED = WorkerResult(role="broken", task="Do something.", error="Timed out")


def test_a_good_review_is_stored() -> None:
    review = review_one(GOOD, FakeReviewerLLM(Review(feedback="Clear, sourced.", score=5)))
    assert review.role == "researcher"
    assert review.score == 5
    assert review.feedback == "Clear, sourced."
    assert review.reviewed


def test_the_model_sees_the_sub_task_and_the_output() -> None:
    fake = FakeReviewerLLM(Review(feedback="ok", score=3))
    review_one(GOOD, fake)
    human_text = fake.received[-1][1]
    assert GOOD.task in human_text and GOOD.output in human_text


def test_failed_worker_is_not_sent_to_the_model() -> None:
    fake = FakeReviewerLLM(Review(feedback="ok", score=3))
    review = review_one(FAILED, fake)
    assert fake.calls == 0
    assert review.score is None and not review.reviewed
    assert "Timed out" in review.feedback


def test_score_out_of_range_is_retried() -> None:
    # A dict with score 9 fails validation (1 to 5 only), so the reviewer asks again.
    fake = FakeReviewerLLM({"feedback": "wow", "score": 9}, Review(feedback="ok", score=4))
    assert review_one(GOOD, fake).score == 4
    assert fake.calls == 2


def test_bad_json_every_time_gives_no_score_but_no_crash() -> None:
    fake = FakeReviewerLLM(OutputParserException("Invalid json output"))
    review = review_one(GOOD, fake)
    assert review.score is None
    assert fake.calls == config.REVIEWER_MAX_ATTEMPTS


def test_ollama_down_gives_no_score_and_a_hint() -> None:
    review = review_one(GOOD, FakeReviewerLLM(ConnectionError("Failed to connect to Ollama")))
    assert review.score is None
    assert "ollama serve" in review.feedback


def test_review_all_keeps_the_order() -> None:
    results = [
        WorkerResult(role=f"w{i}", task="t", output=f"output {i}") for i in range(4)
    ] + [FAILED]
    reviews = review_all(results, llm=FakeReviewerLLM(Review(feedback="ok", score=3)))
    assert [r.role for r in reviews] == ["w0", "w1", "w2", "w3", "broken"]
    assert [r.score for r in reviews] == [3, 3, 3, 3, None]


def test_review_all_with_only_failures_needs_no_model() -> None:
    # llm=None would normally build a real client; with only failures it must not need one.
    reviews = review_all([FAILED])
    assert reviews[0].score is None


def test_real_reviewer_client_can_be_built_without_calling_ollama() -> None:
    assert build_reviewer_llm() is not None


@pytest.mark.live
@pytest.mark.skipif(not config.ollama_is_running(), reason="Ollama is not running.")
def test_live_reviewer_scores_good_output_higher_than_bad() -> None:
    good = WorkerResult(
        role="researcher",
        task="Find the height of Mount Everest in metres, with a source.",
        output="Mount Everest is 8,849 m tall (source: https://en.wikipedia.org/wiki/Mount_Everest).",
    )
    bad = good.model_copy(update={"output": "Bananas are yellow."})
    good_review, bad_review = review_all([good, bad])
    assert good_review.reviewed and bad_review.reviewed
    assert good_review.score > bad_review.score
