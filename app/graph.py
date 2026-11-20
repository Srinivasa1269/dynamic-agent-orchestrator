"""
graph.py - Connects all steps and runs workers in parallel. (Phase 3-4)

What this file does, in simple words:
- A "graph" is a flow chart the program follows. Each box is a step ("node").
- Our flow chart so far:

      START -> planner -> worker, worker, worker ... (all at once) -> END

- The planner decides how many workers there are, so the number of worker
  boxes is not fixed. It is decided while the program runs. That is the
  "dynamic" part of this project.
- Phase 4 will add two more boxes after the workers: reviewer and writer.

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
from app.spawner import SpawnerError
from app.workers import WorkerResult, run_worker


# --- The shared notebook ----------------------------------------------------
class OrchestratorState(TypedDict, total=False):
    """Everything the steps share. Each step reads it and adds to it."""

    task: str  # what the user asked for
    plan: Plan  # written by the planner
    # Every worker adds one result. "operator.add" tells the graph to join
    # the lists together instead of letting workers overwrite each other.
    results: Annotated[list[WorkerResult], operator.add]


class WorkerInput(TypedDict):
    """What one worker box receives."""

    spec: WorkerSpec


# --- Building the flow chart ------------------------------------------------
def build_graph(
    planner_llm: Any = None,
    agent_factory: Callable[[WorkerSpec], Any] | None = None,
) -> Any:
    """Build the flow chart and return it, ready to run.

    Leave both arguments empty in normal use. Tests pass fakes:
    planner_llm:   a fake model for the planner.
    agent_factory: a function that returns a fake agent for a worker.
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

    builder = StateGraph(OrchestratorState)
    builder.add_node("planner", planner_node)
    builder.add_node("worker", worker_node)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", send_to_workers, ["worker"])
    builder.add_edge("worker", END)
    return builder.compile()


# --- The one function other files call --------------------------------------
def run_task(
    task: str,
    planner_llm: Any = None,
    agent_factory: Callable[[WorkerSpec], Any] | None = None,
) -> OrchestratorState:
    """Plan the task, run all workers in parallel, and return everything.

    Returns a dictionary with "task", "plan" and "results". The results are
    in the same order as the workers in the plan.
    Raises PlannerError if no plan could be made.
    """
    graph = build_graph(planner_llm=planner_llm, agent_factory=agent_factory)
    final_state = graph.invoke({"task": task, "results": []})

    # Put results back into plan order, whatever order the workers finished in.
    order = {spec.role: index for index, spec in enumerate(final_state["plan"].workers)}
    final_state["results"] = sorted(
        final_state.get("results", []), key=lambda result: order.get(result.role, 0)
    )
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
    for item in state["results"]:
        status = "OK" if item.succeeded else "FAILED"
        print(f"=== {item.role} [{status}, {item.seconds}s] ===")
        print(item.output if item.succeeded else item.error)
        print()
