"""
workers.py - What a single worker agent does. (Phase 3)

What this file does, in simple words:
- It takes one worker (its role, its sub-task and its agent) and runs it.
- It always hands back a WorkerResult, even when things go wrong:
    * the worker finished      -> the result holds its output
    * the worker took too long -> the result holds a "timed out" error
    * the worker crashed       -> the result holds the error message
- Because run_worker never raises an error, one broken worker can never stop
  the other workers or the final report.
"""

import threading
import time
from typing import Any

from pydantic import BaseModel

from app import config
from app.planner import WorkerSpec


class WorkerResult(BaseModel):
    """What one worker produced."""

    role: str
    task: str
    output: str = ""  # the worker's findings (empty if it failed)
    error: str = ""  # what went wrong (empty if it succeeded)
    seconds: float = 0.0  # how long the worker ran

    @property
    def succeeded(self) -> bool:
        return not self.error


def _final_text(agent_answer: Any) -> str:
    """Pull the worker's last message out of the agent's answer."""
    messages = agent_answer.get("messages", []) if isinstance(agent_answer, dict) else []
    if not messages:
        return ""
    # .text gives only the written answer, leaving out any "thinking" parts.
    return str(getattr(messages[-1], "text", "")).strip()


def run_worker(
    spec: WorkerSpec,
    agent: Any,
    timeout_seconds: float | None = None,
) -> WorkerResult:
    """Run one worker agent on its sub-task and return a WorkerResult.

    spec:  the worker's role and sub-task.
    agent: the agent built by spawner.spawn_worker().
    timeout_seconds: how long to wait. Defaults to WORKER_TIMEOUT_SECONDS.
    """
    if timeout_seconds is None:
        timeout_seconds = config.WORKER_TIMEOUT_SECONDS

    # The agent runs in a helper thread so that we can stop WAITING for it.
    # (Python cannot force a thread to stop. A timed-out worker is left to
    # finish in the background and its answer is thrown away. "daemon=True"
    # means it will not keep the program open.)
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["answer"] = agent.invoke(
                {"messages": [{"role": "user", "content": spec.task}]},
                # Caps how many think/tool steps the agent may take.
                {"recursion_limit": config.WORKER_MAX_STEPS},
            )
        except Exception as error:  # noqa: BLE001 - any failure becomes a result
            box["error"] = error

    started = time.monotonic()
    thread = threading.Thread(target=work, name=f"worker-{spec.role}", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    seconds = round(time.monotonic() - started, 2)

    result = WorkerResult(role=spec.role, task=spec.task, seconds=seconds)

    if thread.is_alive():
        result.error = f"Timed out after {timeout_seconds:g} seconds."
    elif "error" in box:
        error = box["error"]
        result.error = f"{type(error).__name__}: {error}"
    else:
        result.output = _final_text(box.get("answer"))
        if not result.output:
            result.error = "The worker finished but returned no text."

    return result
