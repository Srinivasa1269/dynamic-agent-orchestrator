"""
config.py - All project settings live here, in one place.

What this file does, in simple words:
- It reads your private settings (like the API key) from the .env file.
  A ".env file" is a small text file holding secrets. It is never shared
  or committed to git.
- It defines the limits for how many agents the system may create.

Every other file imports its settings from here, so to change a limit
you only need to edit this one file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# The project folder is one level above this file (app/ -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Read the .env file in the project folder and copy its values into
# environment variables (settings the operating system keeps for a program).
# override=True means the .env file wins over a value already set in Windows,
# so an old key left in your system settings cannot silently be used instead.
# (If there is no .env file, e.g. on a server, the system values are used.)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# The placeholder text in .env.example. It does not count as a real key.
_PLACEHOLDER_KEY = "your-key-here"
DEFAULT_MODEL_NAME = "claude-sonnet-5-5"

# --- Secrets (read from .env, never written in code) -----------------------
# .strip() removes accidental spaces or line breaks around the key.
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()

# --- Model ------------------------------------------------------------------
# Which Claude model to use. You can override it in .env with MODEL_NAME=...
# "or" makes a blank MODEL_NAME= line fall back to the default too.
MODEL_NAME: str = os.getenv("MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME

# --- Agent limits -----------------------------------------------------------
MIN_AGENTS: int = 1  # the planner must always create at least 1 worker
MAX_AGENTS: int = 6  # the planner may never create more than 6 workers

# --- Planner ----------------------------------------------------------------
# The most text (measured in "tokens", roughly word pieces) the planner may
# write, including its private thinking. A plan is short, so this is plenty.
PLANNER_MAX_TOKENS: int = 8000
# If the planner takes longer than this, we stop waiting and report an error.
PLANNER_TIMEOUT_SECONDS: int = 120

# --- Safety -----------------------------------------------------------------
# If one worker takes longer than this, we stop waiting and record an error.
WORKER_TIMEOUT_SECONDS: int = 60


def has_api_key() -> bool:
    """Return True if a real API key was found. Never returns the key itself.

    The placeholder text from .env.example does not count as a real key.
    """
    return bool(ANTHROPIC_API_KEY) and ANTHROPIC_API_KEY != _PLACEHOLDER_KEY


def api_key_looks_valid() -> bool:
    """Return True if the key starts like an Anthropic key ("sk-ant-").

    Returns only True/False so the key itself never appears in test output.
    """
    return ANTHROPIC_API_KEY.startswith("sk-ant-")
