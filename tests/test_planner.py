"""
test_planner.py - Checks the planner without calling the real Claude.

Most tests here use a "fake model": a tiny stand-in that returns a plan we
wrote ourselves. That way the tests are free, fast, and need no API key.
They check OUR code (the limits, the clean-up), not Claude's cleverness.

The last test is marked "live". It calls the real Claude, so it costs a
little money and is skipped by default. Run it on purpose with:
    .venv\\Scripts\\python -m pytest -m live -v
"""

import pytest

from app import config
from app.planner import Plan, PlannerError, WorkerSpec, build_planner_llm, make_plan


class FakeLLM:
    """Pretends to be Claude: always returns the plan it was given."""

    def __init__(self, answer):
        self.answer = answer
        self.received = None  # remembers what make_plan sent us

    def invoke(self, messages):
        self.received = messages
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _plan_with(count: int) -> Plan:
    workers = [WorkerSpec(role=f"worker_{i}", task=f"Do part {i}") for i in range(count)]
    return Plan(reasoning="Because.", workers=workers)


def test_good_plan_is_returned_unchanged() -> None:
    plan = make_plan("Compare A and B", llm=FakeLLM(_plan_with(3)))
    assert [w.role for w in plan.workers] == ["worker_0", "worker_1", "worker_2"]
    assert plan.reasoning == "Because."


def test_task_is_sent_to_the_model() -> None:
    fake = FakeLLM(_plan_with(1))
    make_plan("  Compare A and B  ", llm=fake)
    human_message = fake.received[-1]
    assert human_message[0] == "human"
    assert "Compare A and B" in human_message[1]


def test_too_many_workers_are_cut_to_the_maximum() -> None:
    plan = make_plan("Big task", llm=FakeLLM(_plan_with(config.MAX_AGENTS + 4)))
    assert len(plan.workers) == config.MAX_AGENTS


def test_zero_workers_falls_back_to_one_general_worker() -> None:
    plan = make_plan("Small task", llm=FakeLLM(_plan_with(0)))
    assert len(plan.workers) == config.MIN_AGENTS
    assert plan.workers[0].task == "Small task"


def test_workers_with_empty_tasks_are_dropped() -> None:
    answer = Plan(
        reasoning="r",
        workers=[
            WorkerSpec(role="real", task="Do something"),
            WorkerSpec(role="empty", task="   "),
        ],
    )
    plan = make_plan("Task", llm=FakeLLM(answer))
    assert [w.role for w in plan.workers] == ["real"]


def test_role_names_are_tidied_and_made_unique() -> None:
    answer = Plan(
        reasoning="r",
        workers=[
            WorkerSpec(role="Market Researcher!", task="a"),
            WorkerSpec(role="market_researcher", task="b"),
            WorkerSpec(role="", task="c"),
        ],
    )
    plan = make_plan("Task", llm=FakeLLM(answer))
    assert [w.role for w in plan.workers] == [
        "market_researcher",
        "market_researcher_2",
        "worker",
    ]


def test_empty_task_is_rejected() -> None:
    with pytest.raises(PlannerError):
        make_plan("   ", llm=FakeLLM(_plan_with(1)))


def test_model_failure_becomes_planner_error() -> None:
    with pytest.raises(PlannerError):
        make_plan("Task", llm=FakeLLM(RuntimeError("network down")))


def test_wrong_answer_type_becomes_planner_error() -> None:
    with pytest.raises(PlannerError):
        make_plan("Task", llm=FakeLLM({"not": "a plan"}))


def test_missing_api_key_gives_a_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    with pytest.raises(PlannerError, match="ANTHROPIC_API_KEY"):
        build_planner_llm()


def test_real_client_can_be_built_without_calling_claude(monkeypatch) -> None:
    # A made-up key. Building the client does not contact Claude.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-fake-for-tests")
    assert build_planner_llm() is not None


@pytest.mark.live
@pytest.mark.skipif(not config.has_api_key(), reason="No real ANTHROPIC_API_KEY in .env.")
def test_live_planner_returns_a_valid_plan() -> None:
    plan = make_plan("Compare electric cars and petrol cars on cost and pollution.")
    assert config.MIN_AGENTS <= len(plan.workers) <= config.MAX_AGENTS
    assert all(w.role and w.task for w in plan.workers)
