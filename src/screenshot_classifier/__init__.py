"""Command-line entry point for the screenshot classifier API."""

from pathlib import Path

import uvicorn


def main() -> None:
    """Run the FastAPI service using the bundled model and labels."""
    api_directory = Path(__file__).resolve().parents[1] / "api"
    uvicorn.run("main:app", app_dir=str(api_directory))
