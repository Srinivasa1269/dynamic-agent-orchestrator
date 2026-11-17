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
