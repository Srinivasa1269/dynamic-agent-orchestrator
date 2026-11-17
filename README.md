# dynamic-agent-orchestrator

A planner agent decides how many worker agents a task needs, creates them at
runtime, runs them in parallel, then a reviewer and a writer produce one report.

Full documentation comes in Phase 7.

## Quick setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # then paste your Anthropic key into .env
.venv\Scripts\python -m pytest -v
```

## Try the planner (Phase 2)

The planner asks Claude to split a task into 1 to 6 workers. It needs a real
key in `.env`.

```powershell
.venv\Scripts\python -m app.planner "Compare electric cars and petrol cars"
```

The normal tests use a fake model, so they are free. One extra test calls the
real Claude and only runs when you ask for it:

```powershell
.venv\Scripts\python -m pytest -m live -v
```
