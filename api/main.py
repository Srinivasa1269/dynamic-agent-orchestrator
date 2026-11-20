"""
main.py - FastAPI web server with GET /health and POST /run. (Phase 5)

Start it from the project folder:
    .venv\\Scripts\\python -m uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config
from app.graph import run_task
from app.planner import PlannerError

app = FastAPI(title="Dynamic Agent Orchestrator")


class RunRequest(BaseModel):
    task: str


@app.get("/health")
def health() -> dict:
    """Is the server up, and can it reach the local Ollama model?"""
    return {
        "status": "ok",
        "model": config.OLLAMA_MODEL,
        "ollama_url": config.OLLAMA_BASE_URL,
        "ollama_running": config.ollama_is_running(),
    }


# A plain "def" (not "async def") makes FastAPI run this in a helper thread,
# so a long run does not block /health or other requests.
@app.post("/run")
def run(request: RunRequest) -> dict:
    """Plan the task, run the workers, review them, and return the final report."""
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="The task is empty.")
    try:
        state = run_task(request.task)
    except PlannerError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - report anything else clearly
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error

    return {
        "task": state["task"],
        "report": state["report"].model_dump(),  # the final answer: read this first
        "plan": state["plan"].model_dump(),
        "results": [
            {**result.model_dump(), "succeeded": result.succeeded}
            for result in state["results"]
        ],
        "reviews": [review.model_dump() for review in state["reviews"]],
    }
