"""Shared utilities: logging, atomic JSON writing, canonical hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """Console + rotating file logging. Safe to call more than once."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("textile")
    root.setLevel(getattr(logging, level.upper()))
    if root.handlers:
        return
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)-14s | %(name)s | %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_dir / "pipeline.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(console)
    root.propagate = False


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to a temp file then os.replace -> readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default, allow_nan=False)
        fh.write("\n")
    os.replace(tmp_path, path)


def canonical_sha256(body: Mapping[str, Any]) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

