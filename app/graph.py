"""
graph.py - Connects all steps and runs workers in parallel. (Phase 3-4)

What this file does, in simple words:
- A "graph" is a flow chart the program follows. Each box is a step ("node").
- Our flow chart:

      START -> planner -> worker, worker, worker ... (all at once)
            -> reviewer -> writer -> END

- The planner decides how many workers there are, so the number of worker
  boxes is not fixed. It is decided while the program runs. That is the
  "dynamic" part of this project.
- The reviewer waits until every worker is done, then scores each output.
- The writer merges everything into one final report.

Try it from the terminal:
    .venv\\Scripts\\python -m app.graph "Compare electric cars and petrol cars"
"""

import operator
import sys
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app import spawner
from app.planner import Plan, PlannerError, WorkerSpec, make_plan
from app.reviewer import WorkerReview, review_all
from app.spawner import SpawnerError
from app.workers import WorkerResult, run_worker
from app.writer import Report, write_report


# --- The shared notebook ----------------------------------------------------
class OrchestratorState(TypedDict, total=False):
    """Everything the steps share. Each step reads it and adds to it."""

    task: str  # what the user asked for
    plan: Plan  # written by the planner
    # Every worker adds one result. "operator.add" tells the graph to join
    # the lists together instead of letting workers overwrite each other.
    results: Annotated[list[WorkerResult], operator.add]
    reviews: list[WorkerReview]  # written by the reviewer, in plan order
    report: Report  # written by the writer


class WorkerInput(TypedDict):
    """What one worker box receives."""

    spec: WorkerSpec


def _in_plan_order(state: OrchestratorState) -> list[WorkerResult]:
    """Workers finish in any order. Put their results back in plan order."""
    order = {spec.role: index for index, spec in enumerate(state["plan"].workers)}
    return sorted(state.get("results", []), key=lambda result: order.get(result.role, 0))


# --- Building the flow chart ------------------------------------------------
def build_graph(
    planner_llm: Any = None,
    agent_factory: Callable[[WorkerSpec], Any] | None = None,
    reviewer_llm: Any = None,
    writer_llm: Any = None,
) -> Any:
    """Build the flow chart and return it, ready to run.

    Leave every argument empty in normal use. Tests pass fakes:
    planner_llm:   a fake model for the planner.
    agent_factory: a function that returns a fake agent for a worker.
    reviewer_llm:  a fake model for the reviewer.
    writer_llm:    a fake model for the writer.
    """

    def planner_node(state: OrchestratorState) -> dict:
        return {"plan": make_plan(state["task"], llm=planner_llm)}

    def send_to_workers(state: OrchestratorState) -> list[Send]:
        # One Send per worker = one worker box per worker, all run at once.
        return [Send("worker", {"spec": spec}) for spec in state["plan"].workers]

    def worker_node(worker_input: WorkerInput) -> dict:
        spec = worker_input["spec"]
        try:
            agent = agent_factory(spec) if agent_factory else spawner.spawn_worker(spec)
        except Exception as error:  # noqa: BLE001 - a failed spawn is a failed worker
            failed = WorkerResult(
                role=spec.role,
                task=spec.task,
                error=f"Could not create this worker: {error}",
            )
            return {"results": [failed]}
        return {"results": [run_worker(spec, agent)]}

    def reviewer_node(state: OrchestratorState) -> dict:
        return {"reviews": review_all(_in_plan_order(state), llm=reviewer_llm)}

    def writer_node(state: OrchestratorState) -> dict:
        report = write_report(
            state["task"], _in_plan_order(state), state.get("reviews", []), llm=writer_llm
        )
        return {"report": report}

    builder = StateGraph(OrchestratorState)
    builder.add_node("planner", planner_node)
    builder.add_node("worker", worker_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", send_to_workers, ["worker"])
    # The reviewer runs once, after ALL the worker boxes have finished.
    builder.add_edge("worker", "reviewer")
    builder.add_edge("reviewer", "writer")
    builder.add_edge("writer", END)
    return builder.compile()


# --- The one function other files call --------------------------------------
def run_task(
    task: str,
    planner_llm: Any = None,
    agent_factory: Callable[[WorkerSpec], Any] | None = None,
    reviewer_llm: Any = None,
    writer_llm: Any = None,
) -> OrchestratorState:
    """Plan the task, run all workers in parallel, review them, write the report.

    Returns a dictionary with "task", "plan", "results", "reviews" and
    "report". Results and reviews are in the same order as the plan.
    Raises PlannerError if no plan could be made.
    """
    graph = build_graph(
        planner_llm=planner_llm,
        agent_factory=agent_factory,
        reviewer_llm=reviewer_llm,
        writer_llm=writer_llm,
    )
    final_state = graph.invoke({"task": task, "results": []})
    final_state["results"] = _in_plan_order(final_state)
    return final_state


# --- Small command-line demo ------------------------------------------------
if __name__ == "__main__":
    # Some Windows terminals cannot show every character (for example emoji).
    # This prints a "?" for those instead of crashing.
    sys.stdout.reconfigure(errors="replace")
    user_task = " ".join(sys.argv[1:])
    try:
        state = run_task(user_task)
    except (PlannerError, SpawnerError) as problem:
        print(f"Error: {problem}")
        sys.exit(1)

    print(f"Plan: {state['plan'].reasoning}\n")
    for item, review in zip(state["results"], state["reviews"]):
        status = "OK" if item.succeeded else "FAILED"
        score = f"score {review.score}/5" if review.reviewed else "not scored"
        print(f"- {item.role} [{status}, {item.seconds}s, {score}] {review.feedback}")

    report = state["report"]
    print("\n========== REPORT ==========\n")
    print(report.text)
    if report.fallback:
        print(f"\n(The writer failed: {report.error})")
