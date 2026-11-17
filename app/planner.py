"""
planner.py - Decides how many worker agents a task needs and their roles. (Phase 2)

What this file does, in simple words:
- You give it a task, for example "Compare electric cars and petrol cars".
- It asks Claude to split that task into a small team of workers.
- It returns a "Plan": a list of workers, each with a role and its own sub-task.

The planner only PLANS. It does not create or run the workers - that is the
job of spawner.py and graph.py (Phase 3).

Safety: Claude's answer is never trusted blindly. After Claude replies, our
own code enforces the limits from config.py (MIN_AGENTS to MAX_AGENTS).

Try it from the terminal:
    .venv\\Scripts\\python -m app.planner "Compare electric cars and petrol cars"
"""

import re
import sys
from typing import Any

from pydantic import BaseModel, Field

from app import config


# --- The shape of a plan ----------------------------------------------------
# These classes describe exactly what we want back from Claude. Claude is
# forced to answer in this shape ("structured output"), so we get real Python
# objects instead of free text we would have to pick apart.


class WorkerSpec(BaseModel):
    """One worker agent the planner wants to create."""

    role: str = Field(
        description=(
            "Short snake_case name for this worker's job, for example "
            "'market_researcher'. Must be different for every worker."
        )
    )
    task: str = Field(
        description=(
            "The exact sub-task this worker must do. It must make sense on "
            "its own, because the worker cannot see the other workers."
        )
    )


class Plan(BaseModel):
    """The planner's full answer: why, and which workers."""

    reasoning: str = Field(
        description="One or two sentences on why this number of workers fits the task."
    )
    workers: list[WorkerSpec] = Field(
        description="The workers to create. They all run at the same time."
    )


class PlannerError(Exception):
    """Raised when the planner cannot produce a plan."""


# --- The instructions we give Claude ----------------------------------------
PLANNER_SYSTEM_PROMPT = f"""\
You are the planner in a multi-agent system. A user gives you a task. You split \
it into a team of worker agents, and after you answer, the system creates one \
agent per worker in your plan and runs them all at the same time. A reviewer \
then scores each worker's output, and a writer merges everything into one \
final report for the user.

How to decide the team:
- Use between {config.MIN_AGENTS} and {config.MAX_AGENTS} workers. Every worker \
costs time and money, so use as few as the task really needs. A simple question \
needs 1 worker. Add a worker only when it covers a clearly separate part of the task.
- Workers run in parallel and cannot see each other's work. Give each one a \
sub-task that can be finished alone, with no overlap between workers and no \
worker waiting on another worker's result.
- Each worker sees only the sub-task you write for it, not the user's original \
task. Put everything it needs into that sub-task: the goal, the scope, and what \
its output should contain.
- Workers can search the web and use a calculator. They cannot do anything else.
- Do not create a worker to review, combine, or summarise the others. The \
reviewer and the writer already do that.
"""


# --- Talking to Claude ------------------------------------------------------
def build_planner_llm() -> Any:
    """Create the Claude client that returns a Plan object.

    The import is inside the function so that tests (which use a fake model)
    never need a real API key or a network connection.
    """
    if not config.has_api_key():
        raise PlannerError(
            "ANTHROPIC_API_KEY is missing. Copy .env.example to .env "
            "and paste your key there."
        )

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=config.MODEL_NAME,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=config.PLANNER_MAX_TOKENS,
        timeout=config.PLANNER_TIMEOUT_SECONDS,
    )
    # method="json_schema" uses Claude's built-in structured output feature,
    # which guarantees the reply matches the Plan shape.
    return llm.with_structured_output(Plan, method="json_schema")


# --- Enforcing the limits ourselves -----------------------------------------
def _clean_role(role: str) -> str:
    """Turn any text into a tidy snake_case name, e.g. 'Market Researcher!' -> 'market_researcher'."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", role.strip().lower()).strip("_")
    return cleaned or "worker"


def _enforce_limits(plan: Plan, task: str) -> Plan:
    """Make the plan obey our rules, whatever Claude answered.

    - Workers with an empty sub-task are dropped.
    - Role names are tidied and made unique (later phases use them as labels).
    - More than MAX_AGENTS workers: the extra ones are cut off.
    - Fewer than MIN_AGENTS workers: one general worker gets the whole task.
    """
    workers: list[WorkerSpec] = []
    used_roles: set[str] = set()

    for worker in plan.workers:
        if not worker.task.strip():
            continue
        base = _clean_role(worker.role)
        role = base
        number = 2
        while role in used_roles:  # e.g. researcher, researcher_2, researcher_3
            role = f"{base}_{number}"
            number += 1
        used_roles.add(role)
        workers.append(WorkerSpec(role=role, task=worker.task.strip()))

    workers = workers[: config.MAX_AGENTS]

    if len(workers) < config.MIN_AGENTS:
        workers = [WorkerSpec(role="general_researcher", task=task)]

    return Plan(reasoning=plan.reasoning.strip(), workers=workers)


# --- The one function other files call --------------------------------------
def make_plan(task: str, llm: Any = None) -> Plan:
    """Ask Claude to plan a team of workers for this task.

    task: what the user wants done.
    llm:  leave empty in normal use. Tests pass a fake model here.

    Returns a Plan with between MIN_AGENTS and MAX_AGENTS workers.
    Raises PlannerError if the task is empty or Claude cannot be reached.
    """
    task = task.strip()
    if not task:
        raise PlannerError("The task is empty. Please describe what you want done.")

    if llm is None:
        llm = build_planner_llm()

    try:
        raw_plan = llm.invoke(
            [
                ("system", PLANNER_SYSTEM_PROMPT),
                ("human", f"Task from the user:\n\n{task}"),
            ]
        )
    except Exception as error:  # network problem, bad key, refusal, bad reply...
        raise PlannerError(f"The planner could not create a plan: {error}") from error

    if not isinstance(raw_plan, Plan):
        raise PlannerError("The planner returned an answer that is not a Plan.")

    return _enforce_limits(raw_plan, task)


# --- Small command-line demo ------------------------------------------------
if __name__ == "__main__":
    user_task = " ".join(sys.argv[1:])
    try:
        result = make_plan(user_task)
    except PlannerError as problem:
        print(f"Error: {problem}")
        sys.exit(1)

    print(f"Why: {result.reasoning}\n")
    print(f"Workers ({len(result.workers)}):")
    for index, spec in enumerate(result.workers, start=1):
        print(f"  {index}. {spec.role}\n     {spec.task}")
