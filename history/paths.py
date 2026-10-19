"""Session layout: runs/<RUN_ID>/{lora/<variant>/, eval/{rollouts,metrics,tables}/}.

RUN_ID comes from the environment; when it is unset, training creates a UTC
timestamp session and exports it so every later stage in the same process
(and child processes) writes to the same session. runs/latest.json points at
the most recent completed session for the eval stages' default.

This module was previously git-ignored (``history/`` in .gitignore), which made
a fresh clone fail at import time. It is reconstructed from its call sites.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def active_session_id() -> str | None:
    value = os.environ.get("RUN_ID", "").strip()
    return value or None


def ensure_session() -> str:
    """Return RUN_ID, creating (and exporting) a timestamp session if unset."""
    run_id = active_session_id()
    if run_id is None:
        run_id = utc_stamp()
        os.environ["RUN_ID"] = run_id
    (config.RUNS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    return run_id


def session_dir(run_id: str | None = None) -> Path:
    return config.RUNS_DIR / (run_id or ensure_session())


def lora_root(run_id: str | None = None) -> Path:
    return session_dir(run_id) / "lora"


def lora_variant_dir(variant: str, run_id: str | None = None) -> Path:
    """Publish directory for a variant: runs/<id>/lora/<variant>."""
    return lora_root(run_id) / variant


def new_lora_variant_dir(variant: str, run_id: str | None = None) -> Path:
    """Training run directory (holds run_info.json and adapter/checkpoint-*)."""
    path = lora_variant_dir(variant, run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def eval_run_dir(run_id: str | None = None) -> Path:
    return session_dir(run_id) / "eval"


def new_eval_dir(run_id: str | None = None) -> Path:
    path = eval_run_dir(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_eval_dir(run_dir: Path) -> Path:
    """Accept runs/<id>, runs/<id>/eval, or a stage folder inside eval/."""
    run_dir = Path(run_dir)
    if run_dir.name == "eval":
        return run_dir
    if (run_dir / "eval").is_dir() or run_dir.parent == config.RUNS_DIR:
        return run_dir / "eval"
    if run_dir.parent.name == "eval":
        return run_dir.parent
    return run_dir


def write_latest_pointer(run_id: str | Path) -> None:
    run_id = Path(run_id).name if isinstance(run_id, Path) else str(run_id)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "updated_at": datetime.now(timezone.utc).isoformat()}
    tmp = config.RUNS_DIR / f"latest.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(config.RUNS_DIR / "latest.json")
