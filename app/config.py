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
load_dotenv(PROJECT_ROOT / ".env")

# --- Secrets (read from .env, never written in code) -----------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# --- Model ------------------------------------------------------------------
# Which Claude model to use. You can override it in .env with MODEL_NAME=...
MODEL_NAME: str = os.getenv("MODEL_NAME", "claude-sonnet-5-5")

# --- Agent limits -----------------------------------------------------------
MIN_AGENTS: int = 1  # the planner must always create at least 1 worker
MAX_AGENTS: int = 6  # the planner may never create more than 6 workers

# --- Safety -----------------------------------------------------------------
# If one worker takes longer than this, we stop waiting and record an error.
WORKER_TIMEOUT_SECONDS: int = 60


def has_api_key() -> bool:
    """Return True if a real API key was found. Never returns the key itself.

    The placeholder text from .env.example does not count as a real key.
    """
    key = ANTHROPIC_API_KEY.strip()
    return bool(key) and key != "your-key-here"
