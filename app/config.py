"""
config.py - All project settings live here, in one place.

What this file does, in simple words:
- It reads your settings from the .env file. A ".env file" is a small text
  file with local settings. It is never committed to git.
- The AI model runs on your own computer with Ollama. No cloud API key is
  needed.
- It defines the limits for how many agents the system may create.

Every other file imports its settings from here, so to change a limit
you only need to edit this one file.
"""

import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# The project folder is one level above this file (app/ -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Read the .env file in the project folder and copy its values into
# environment variables. override=True means the .env file wins over an old
# value left in your Windows settings.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _text_setting(name: str, default: str) -> str:
    """Read a text setting from .env. A blank value falls back to the default."""
    return os.getenv(name, "").strip() or default


def _number_setting(name: str, default: int) -> int:
    """Read a whole-number setting from .env. A blank or bad value uses the default."""
    try:
        return int(os.getenv(name, "").strip())
    except ValueError:
        return default


# --- Model (Ollama, running locally) ---------------------------------------
# Which Ollama model to use. It must support tool calling. See the list of
# your models with:  ollama list
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_MODEL: str = _text_setting("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
# Where the Ollama server is listening. This is Ollama's normal address.
OLLAMA_BASE_URL: str = _text_setting("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# How much text (in "tokens", roughly word pieces) the model can see at once.
# Workers read search results, so they need more than Ollama's small default.
OLLAMA_NUM_CTX: int = _number_setting("OLLAMA_NUM_CTX", 8192)

# --- Agent limits -----------------------------------------------------------
MIN_AGENTS: int = 1  # the planner must always create at least 1 worker
MAX_AGENTS: int = 6  # the planner may never create more than 6 workers

# --- Planner ----------------------------------------------------------------
# The most tokens the planner may write. A plan is short, so this is plenty.
PLANNER_MAX_TOKENS: int = 1024
# If the planner takes longer than this, we stop waiting and report an error.
# A local model on a laptop CPU is slow, so this is generous.
PLANNER_TIMEOUT_SECONDS: int = _number_setting("PLANNER_TIMEOUT_SECONDS", 300)
# How many times we ask again if the planner's answer is not a valid plan.
PLANNER_MAX_ATTEMPTS: int = 3

# --- Workers ----------------------------------------------------------------
# The most tokens one worker may write in a single reply.
WORKER_MAX_TOKENS: int = 2048
# The most think/tool steps one worker may take before it is stopped.
WORKER_MAX_STEPS: int = 25
# How many results one web search returns.
SEARCH_MAX_RESULTS: int = 5

# --- Safety -----------------------------------------------------------------
# If one worker takes longer than this, we stop waiting and record an error.
# All workers share one local model, so they often wait in a queue.
WORKER_TIMEOUT_SECONDS: int = _number_setting("WORKER_TIMEOUT_SECONDS", 600)


def ollama_is_running(timeout_seconds: float = 2.0) -> bool:
    """Return True if the Ollama server answers at OLLAMA_BASE_URL."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/version", timeout=timeout_seconds):
            return True
    except OSError:  # not running, wrong address, refused...
        return False
