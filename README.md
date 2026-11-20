# dynamic-agent-orchestrator

A planner agent decides how many worker agents a task needs, creates them at
runtime, runs them in parallel, then a reviewer and a writer produce one report.

Everything runs **on your own computer** with [Ollama](https://ollama.com).
No cloud API key is needed. (Web search still uses the internet.)

Full documentation comes in Phase 7.

## 1. Install Ollama and a model

Install Ollama, then get a model that supports **tool calling**:

```powershell
ollama pull llama3.1      # about 4.9 GB; skip if "ollama list" already shows it
ollama list               # check it is there
```

Ollama usually starts by itself. If it is not running, start it with
`ollama serve`.

## 2. Set up the project (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env    # optional: change the model or timeouts here
.venv\Scripts\python -m pytest -v
```

Settings in `.env` (all optional):

| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:latest` | The Ollama model to use (must support tools) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is listening |
| `OLLAMA_NUM_CTX` | `8192` | How much text the model can read at once |
| `PLANNER_TIMEOUT_SECONDS` | `300` | How long the planner may take |
| `WORKER_TIMEOUT_SECONDS` | `600` | How long one worker may take |
| `REVIEWER_TIMEOUT_SECONDS` | `300` | How long one review may take |
| `WRITER_TIMEOUT_SECONDS` | `600` | How long writing the report may take |

## How a task flows

```
/run -> planner -> worker, worker, ... (all at once) -> reviewer -> writer -> report
```

1. **Planner** splits the task into 1 to 6 workers.
2. **Workers** each do one part. They can search the web and use a calculator.
3. **Reviewer** scores each worker's output from 1 to 5, with a short reason.
4. **Writer** merges everything into one report, trusting higher scores more.
   If the writer fails, you get the workers' own outputs instead.

## 3. Run the web server

```powershell
.venv\Scripts\python -m uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/docs to try it in the browser, or:

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/run -H "Content-Type: application/json" -d "{\"task\": \"Write a short plan to learn Docker in 7 days\"}"
```

`/run` returns `report` (the final answer: read `report.text`), plus the
`plan`, every worker's `results`, and the `reviews`. If `report.fallback` is
`true`, the writer failed and the text is the workers' raw outputs.
On a laptop CPU, one run takes about 3 to 6 minutes.

## Try the parts from the terminal

```powershell
# Only the planner: splits a task into 1 to 6 workers
.venv\Scripts\python -m app.planner "Compare electric cars and petrol cars"

# The whole flow: planner, workers, reviewer, writer. Prints the scores and the report.
.venv\Scripts\python -m app.graph "Compare electric cars and petrol cars"
```

## Tests

The normal tests use fake models. They need no Ollama, no internet and no key.
A few extra "live" tests use the real local model or the real web search, and
only run when you ask for them (Ollama must be running):

```powershell
.venv\Scripts\python -m pytest -m live -v
```

## Limits of a local model

- **Slower:** on a laptop CPU, a plan takes around 30–60 s, one worker 1–3
  minutes, each review about 15–30 s, and the report 1–2 minutes. Workers
  share one model, so they often wait in a queue.
- **Weaker:** an 8B model writes simpler answers than large cloud models,
  sometimes forgets to cite URLs, and may pick more workers than needed. Our
  code checks every plan, retries bad ones, and enforces the 1–6 worker limit.
- **Sources can be made up:** a worker may write a URL from memory instead
  of from a search result. The writer keeps only URLs the workers gave, but
  it cannot check them. Open a link before you trust it.
