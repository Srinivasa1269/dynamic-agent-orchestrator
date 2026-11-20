"""
spawner.py - Creates one worker agent per role in the plan. (Phase 3)

What this file does, in simple words:
- The planner hands us a list of workers (a role and a sub-task each).
- For every worker we build an "agent": the local Ollama model plus the
  tools from tools.py
  plus instructions written for that worker's role.
- An agent can think, call a tool, read the result, and repeat until its
  sub-task is done.

This file only CREATES agents. Running one is the job of workers.py.
"""

from typing import Any

from app import config
from app.planner import Plan, WorkerSpec
from app.tools import WORKER_TOOLS


class SpawnerError(Exception):
    """Raised when a worker agent cannot be created."""


def build_worker_prompt(spec: WorkerSpec) -> str:
    """Write the instructions for one worker, based on its role."""
    return f"""\
You are a worker agent in a multi-agent system. Your role is: {spec.role}.

A planner split a user's task into parts and gave you one part. Other workers \
are handling the other parts at the same time; you cannot see their work and \
they cannot see yours. When you finish, a reviewer scores your output and a \
writer merges it with the others into one report for the user. The writer sees \
only your final message, so it must stand on its own.

How to work:
- Do only the sub-task you are given. Do not try to cover the other parts.
- Use web_search for facts, figures and anything recent, rather than relying \
on memory. Use calculator for every calculation.
- If a search fails or finds nothing useful, try different words. If you still \
cannot find something, say so plainly instead of guessing.

Your final message should contain your findings, written clearly, with the \
URL of the source next to each fact you took from the web. State anything you \
could not find or are unsure about.
"""


def build_worker_llm() -> Any:
    """Create the Ollama client that workers use.

    Building the client does not contact Ollama, so tests can call this
    even when Ollama is not running.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,  # low = sticks to facts and calls tools more reliably
        num_ctx=config.OLLAMA_NUM_CTX,  # room for the search results it reads
        num_predict=config.WORKER_MAX_TOKENS,
        client_kwargs={"timeout": config.WORKER_TIMEOUT_SECONDS},
    )


def spawn_worker(spec: WorkerSpec, llm: Any = None) -> Any:
    """Create the agent for one worker.

    spec: the worker's role and sub-task, from the planner.
    llm:  leave empty in normal use. Tests pass a fake model here.
    """
    from langchain.agents import create_agent

    if llm is None:
        llm = build_worker_llm()

    return create_agent(
        model=llm,
        tools=WORKER_TOOLS,
        system_prompt=build_worker_prompt(spec),
        name=spec.role,
    )


def spawn_workers(plan: Plan, llm: Any = None) -> list[tuple[WorkerSpec, Any]]:
    """Create one agent for every worker in the plan.

    Returns a list of (spec, agent) pairs, in the same order as the plan.
    All agents share one Ollama client.
    """
    if llm is None:
        llm = build_worker_llm()
    return [(spec, spawn_worker(spec, llm)) for spec in plan.workers]
