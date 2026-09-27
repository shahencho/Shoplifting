"""Load config.yaml and .env. All paths are resolved relative to the repo root."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    load_dotenv(ROOT / ".env")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def vlm_settings() -> dict:
    load_dotenv(ROOT / ".env")
    return {
        "base_url": os.getenv("VLM_API_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.getenv("VLM_API_KEY", ""),
        "model": os.getenv("VLM_MODEL_NAME", ""),
    }
