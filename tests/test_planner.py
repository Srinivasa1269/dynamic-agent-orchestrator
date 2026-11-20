"""
test_planner.py - Checks the planner without calling the real model.

Most tests here use a "fake model": a tiny stand-in that returns a plan we
wrote ourselves. That way the tests are fast and need no Ollama, no internet
and no API key. They check OUR code (the limits, the clean-up, the retries),
not the model's cleverness.

The last test is marked "live". It calls the real local model through
Ollama and is skipped by default. Run it on purpose with:
    .venv\\Scripts\\python -m pytest -m live -v
"""

import pytest
from langchain_core.exceptions import OutputParserException

from app import config
from app.planner import Plan, PlannerError, WorkerSpec, build_planner_llm, make_plan


class FakeLLM:
    """Pretends to be the model.

    Give it one answer to return every time, or a list of answers to return
    one after another (to test retries). An Exception answer is raised.
    """

    def __init__(self, *answers):
        self.answers = list(answers)
        self.received = None  # remembers what make_plan sent last
        self.calls = 0

    def invoke(self, messages):
        self.received = messages
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return answer


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


def test_prompt_shows_the_model_the_json_format() -> None:
    fake = FakeLLM(_plan_with(1))
    make_plan("Task", llm=fake)
    system_prompt = fake.received[0][1]
    assert '"workers"' in system_prompt and '"role"' in system_prompt


def test_plan_given_as_a_dict_is_accepted() -> None:
    answer = {"reasoning": "r", "workers": [{"role": "a", "task": "Do it"}]}
    plan = make_plan("Task", llm=FakeLLM(answer))
    assert [w.role for w in plan.workers] == ["a"]


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


# --- Retries ----------------------------------------------------------------
def test_bad_json_is_retried_and_then_succeeds() -> None:
    fake = FakeLLM(OutputParserException("Invalid json output"), _plan_with(2))
    plan = make_plan("Task", llm=fake)
    assert len(plan.workers) == 2
    assert fake.calls == 2
    # The retry tells the model what was wrong.
    assert "not a valid plan" in fake.received[-1][1]


def test_dict_with_missing_fields_is_retried() -> None:
    fake = FakeLLM({"workers": "oops"}, _plan_with(1))
    assert len(make_plan("Task", llm=fake).workers) == 1
    assert fake.calls == 2


def test_gives_up_after_the_maximum_number_of_tries() -> None:
    fake = FakeLLM({"not": "a plan"})
    with pytest.raises(PlannerError, match="no valid plan"):
        make_plan("Task", llm=fake)
    assert fake.calls == config.PLANNER_MAX_ATTEMPTS


def test_model_failure_is_not_retried_and_becomes_planner_error() -> None:
    fake = FakeLLM(RuntimeError("something broke"))
    with pytest.raises(PlannerError, match="something broke"):
        make_plan("Task", llm=fake)
    assert fake.calls == 1


def test_connection_failure_explains_how_to_start_ollama() -> None:
    fake = FakeLLM(ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(PlannerError, match="ollama serve"):
        make_plan("Task", llm=fake)


# --- Real client ------------------------------------------------------------
def test_real_client_can_be_built_without_calling_ollama() -> None:
    # Building the client does not contact Ollama, so this works offline.
    assert build_planner_llm() is not None


@pytest.mark.live
@pytest.mark.skipif(not config.ollama_is_running(), reason="Ollama is not running.")
def test_live_planner_returns_a_valid_plan() -> None:
    plan = make_plan("Compare electric cars and petrol cars on cost and pollution.")
    assert config.MIN_AGENTS <= len(plan.workers) <= config.MAX_AGENTS
    assert all(w.role and w.task for w in plan.workers)
