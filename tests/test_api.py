"""
test_api.py - Checks the web server without running the real model.

TestClient sends pretend web requests straight to our FastAPI app.
run_task is swapped for a fake, so no Ollama or internet is needed.
"""

from fastapi.testclient import TestClient

import api.main as server
from app.planner import Plan, PlannerError, WorkerSpec
from app.workers import WorkerResult

client = TestClient(server.app)


def _fake_run_task(task: str) -> dict:
    spec = WorkerSpec(role="researcher", task="Find facts.")
    return {
        "task": task,
        "plan": Plan(reasoning="One worker is enough.", workers=[spec]),
        "results": [
            WorkerResult(role="researcher", task="Find facts.", output="Facts.", seconds=1.5),
            WorkerResult(role="broken", task="Fail.", error="boom"),
        ],
    }


def test_health_reports_ok_and_the_model(monkeypatch) -> None:
    monkeypatch.setattr(server.config, "ollama_is_running", lambda: True)
    answer = client.get("/health")
    assert answer.status_code == 200
    body = answer.json()
    assert body["status"] == "ok"
    assert body["ollama_running"] is True
    assert body["model"] == server.config.OLLAMA_MODEL


def test_run_returns_plan_and_results_as_json(monkeypatch) -> None:
    monkeypatch.setattr(server, "run_task", _fake_run_task)
    answer = client.post("/run", json={"task": "Learn Docker"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["task"] == "Learn Docker"
    assert body["plan"]["workers"][0]["role"] == "researcher"
    assert body["results"][0]["output"] == "Facts."
    assert body["results"][0]["succeeded"] is True
    assert body["results"][1]["succeeded"] is False


def test_run_rejects_an_empty_task() -> None:
    assert client.post("/run", json={"task": "   "}).status_code == 400


def test_run_reports_planner_problems(monkeypatch) -> None:
    def broken(task: str):
        raise PlannerError("Ollama is not running")

    monkeypatch.setattr(server, "run_task", broken)
    answer = client.post("/run", json={"task": "Learn Docker"})
    assert answer.status_code == 502
    assert "Ollama" in answer.json()["detail"]
