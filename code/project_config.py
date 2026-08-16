"""
Digital Detective — Shared runtime configuration helpers
=========================================================
Loads environment variables from a local .env file (if present) and
exposes small helpers used by every phase script. Centralizing this
here means no phase script ever hardcodes a secret.

Setup:
    1. Copy .env.example to .env
    2. Fill in GROQ_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    3. Never commit .env to git (it's already in .gitignore)
"""

import os
from pathlib import Path


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env if the file exists.

    Silently does nothing if .env is missing — scripts fall back to
    get_required_env() raising a clear error only when a value is
    actually needed.
    """
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load as soon as this module is imported by any phase script
load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    """Return a trimmed environment variable, or default if unset/empty."""
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip()


def get_required_env(name: str) -> str:
    """Return a required env var, or raise a clear, actionable error."""
    value = get_env(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Create a .env file from .env.example and set {name}."
        )
    return value
