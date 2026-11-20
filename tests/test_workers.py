"""
test_workers.py - Checks workers, the spawner and the graph with fake agents.

No test here calls the real model, except the one marked "live" at the end.
A "fake agent" is a tiny stand-in with the same invoke() function a real
agent has, so we can test success, crashes and timeouts on purpose.
"""

import threading
import time

import pytest
from langchain_core.messages import AIMessage

from app import config, graph
from app.planner import Plan, PlannerError, WorkerSpec
from app.reviewer import Review
from app.spawner import SpawnerError, build_worker_llm, build_worker_prompt, spawn_worker
from app.workers import run_worker


class FakeAgent:
    """Pretends to be a worker agent."""

    def __init__(self, answer="Findings.", delay: float = 0.0, error: Exception | None = None):
        self.answer = answer
        self.delay = delay
        self.error = error
        self.received_task = None

    def invoke(self, agent_input, run_config=None):
        self.received_task = agent_input["messages"][0]["content"]
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return {"messages": [AIMessage(content=self.answer)]}


class FakePlannerLLM:
    """Pretends to be the planner's model: always returns the same plan."""

    def __init__(self, plan: Plan):
        self.plan = plan

    def invoke(self, messages):
        return self.plan


class FakeReviewerLLM:
    """Pretends to be the reviewer's model: every output gets a 4."""

    def invoke(self, messages):
        return Review(feedback="Fine.", score=4)


class FakeWriterLLM:
    """Pretends to be the writer's model: returns a fixed report."""

    def invoke(self, messages):
        return AIMessage(content="Final report.")


def run_task(task: str, **fakes):
    """The real run_task, but with fake reviewer and writer models (no Ollama)."""
    fakes.setdefault("reviewer_llm", FakeReviewerLLM())
    fakes.setdefault("writer_llm", FakeWriterLLM())
    return graph.run_task(task, **fakes)


SPEC =WorkerSpec(role="researcher", task="Find the height of Mount Everest.")


def _plan(*roles: str) -> Plan:
    return Plan(
        reasoning="r",
        workers=[WorkerSpec(role=role, task=f"Task for {role}") for role in roles],
    )


# --- run_worker -------------------------------------------------------------
def test_worker_returns_the_agents_final_text() -> None:
    agent = FakeAgent(answer="Everest is 8,849 m.")
    result = run_worker(SPEC, agent)
    assert result.succeeded
    assert result.output == "Everest is 8,849 m."
    assert result.role == "researcher"
    assert agent.received_task == SPEC.task


def test_worker_reads_text_from_a_reply_that_also_contains_thinking() -> None:
    blocks = [
        {"type": "thinking", "thinking": "hmm", "signature": "s"},
        {"type": "text", "text": "The answer."},
    ]
    result = run_worker(SPEC, FakeAgent(answer=blocks))
    assert result.output == "The answer."


def test_worker_crash_becomes_an_error_result() -> None:
    result = run_worker(SPEC, FakeAgent(error=RuntimeError("boom")))
    assert not result.succeeded
    assert "boom" in result.error
    assert result.output == ""


def test_worker_timeout_becomes_an_error_result() -> None:
    started = time.monotonic()
    result = run_worker(SPEC, FakeAgent(delay=2.0), timeout_seconds=0.2)
    assert time.monotonic() - started < 1.5  # we stopped waiting early
    assert not result.succeeded
    assert "Timed out" in result.error


def test_worker_with_empty_answer_is_an_error() -> None:
    result = run_worker(SPEC, FakeAgent(answer="   "))
    assert not result.succeeded


# --- spawner ----------------------------------------------------------------
def test_worker_prompt_mentions_the_role() -> None:
    assert "researcher" in build_worker_prompt(SPEC)


def test_worker_llm_uses_the_configured_ollama_model() -> None:
    llm = build_worker_llm()
    assert llm.model == config.OLLAMA_MODEL
    assert llm.base_url == config.OLLAMA_BASE_URL


def test_real_agent_can_be_built_without_calling_ollama() -> None:
    # Building the agent does not contact Ollama, so this works offline.
    agent = spawn_worker(SPEC)
    assert hasattr(agent, "invoke")


# --- graph ------------------------------------------------------------------
def test_graph_runs_one_worker_per_planned_role() -> None:
    state = run_task(
        "Some task",
        planner_llm=FakePlannerLLM(_plan("alpha", "beta", "gamma")),
        agent_factory=lambda spec: FakeAgent(answer=f"Output of {spec.role}"),
    )
    assert [r.role for r in state["results"]] == ["alpha", "beta", "gamma"]
    assert [r.output for r in state["results"]] == [
        "Output of alpha",
        "Output of beta",
        "Output of gamma",
    ]


def test_graph_runs_workers_at_the_same_time() -> None:
    # Four workers that each take 0.5s. One after another = 2s. Together = ~0.5s.
    running_together = []
    lock = threading.Lock()
    active = 0

    class SlowAgent(FakeAgent):
        def invoke(self, agent_input, run_config=None):
            nonlocal active
            with lock:
                active += 1
                running_together.append(active)
            time.sleep(0.5)
            with lock:
                active -= 1
            return {"messages": [AIMessage(content="done")]}

    started = time.monotonic()
    state = run_task(
        "Some task",
        planner_llm=FakePlannerLLM(_plan("a", "b", "c", "d")),
        agent_factory=lambda spec: SlowAgent(),
    )
    assert len(state["results"]) == 4
    assert max(running_together) > 1
    assert time.monotonic() - started < 1.5


def test_one_failing_worker_does_not_stop_the_others() -> None:
    def factory(spec: WorkerSpec):
        if spec.role == "bad":
            return FakeAgent(error=RuntimeError("boom"))
        return FakeAgent(answer="fine")

    state = run_task(
        "Some task",
        planner_llm=FakePlannerLLM(_plan("good", "bad", "also_good")),
        agent_factory=factory,
    )
    by_role = {r.role: r for r in state["results"]}
    assert by_role["good"].succeeded and by_role["also_good"].succeeded
    assert not by_role["bad"].succeeded


def test_worker_that_cannot_be_created_is_recorded_as_failed() -> None:
    def factory(spec: WorkerSpec):
        raise SpawnerError("no key")

    state = run_task(
        "Some task",
        planner_llm=FakePlannerLLM(_plan("only")),
        agent_factory=factory,
    )
    assert "Could not create" in state["results"][0].error


def test_graph_passes_planner_errors_up() -> None:
    with pytest.raises(PlannerError):
        run_task("   ", planner_llm=FakePlannerLLM(_plan("a")))


def test_graph_reviews_every_worker_and_writes_a_report() -> None:
    def factory(spec: WorkerSpec):
        # "slow" finishes last, to check reviews still come back in plan order.
        if spec.role == "slow":
            return FakeAgent(answer="late", delay=0.3)
        if spec.role == "bad":
            return FakeAgent(error=RuntimeError("boom"))
        return FakeAgent(answer="fine")

    state = run_task(
        "Some task",
        planner_llm=FakePlannerLLM(_plan("slow", "bad", "fast")),
        agent_factory=factory,
    )
    assert [r.role for r in state["reviews"]] == ["slow", "bad", "fast"]
    assert [r.score for r in state["reviews"]] == [4, None, 4]  # failed = not scored
    assert state["report"].text == "Final report."
    assert not state["report"].fallback


# --- live -------------------------------------------------------------------
@pytest.mark.live
@pytest.mark.skipif(not config.ollama_is_running(), reason="Ollama is not running.")
def test_live_full_run() -> None:
    # graph.run_task (not the wrapper above) so the real reviewer and writer run.
    state = graph.run_task("What is 17% of 2,340, and what is the capital of Australia?")
    assert any(r.succeeded for r in state["results"])
    assert len(state["reviews"]) == len(state["results"])
    assert any(review.reviewed for review in state["reviews"])
    assert state["report"].text.strip()
    assert not state["report"].fallback
