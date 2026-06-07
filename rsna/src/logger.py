"""
src/logger.py
=============
Created first, before anything else.

Sets up two handlers:
  1. FileHandler  → outputs/logs/{YYYY-MM-DD}_{script_name}.log   (human-readable)
  2. FileHandler  → outputs/logs/run_log.jsonl                     (structured JSON, one dict/line)

Also maintains:
  outputs/logs/run_summary.csv  — one row per training run

Usage (at the top of every script):
    from src.logger import get_logger, log_event, write_run_summary
    logger = get_logger(run_id="run_20240101_120000", script_name="module1")
    log_event(run_id, "module1", "script_start", config={"batch": 16})
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from config.settings import CFG

# Ensure log directory exists at import time
CFG.LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG_PATH     = CFG.LOGS_DIR / "run_log.jsonl"
RUN_SUMMARY_PATH = CFG.LOGS_DIR / "run_summary.csv"

RUN_SUMMARY_COLUMNS = [
    "run_id", "model_name", "start_time", "end_time",
    "best_epoch", "best_val_loss", "test_loss", "notes",
]


# ── public API ────────────────────────────────────────────────────────────────

def get_logger(run_id: str, script_name: str) -> logging.Logger:
    """
    Return a logger with:
      - Console handler (INFO+)
      - Dated .log file handler (DEBUG+)
    Every line includes timestamp, level, run_id, and script_name.
    """
    date_str  = datetime.now().strftime("%Y-%m-%d")
    log_file  = CFG.LOGS_DIR / f"{date_str}_{script_name}.log"
    logger_id = f"{run_id}:{script_name}"
    logger    = logging.getLogger(logger_id)

    if logger.handlers:          # idempotent — safe to call multiple times
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler 1 — human-readable .log file
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Handler 2 — console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def log_event(run_id: str, script_name: str, event: str, **kwargs) -> None:
    """
    Append one JSON line to run_log.jsonl.
    Every record contains: timestamp, run_id, script_name, event, plus any
    keyword arguments supplied by the caller.

    Examples
    --------
    log_event(run_id, "module1", "script_start", config=cfg_dict)
    log_event(run_id, "module4", "epoch_end", epoch=5, val_loss=0.42)
    log_event(run_id, "module4", "early_stop", patience_exhausted=True)
    """
    record = {
        "timestamp":   datetime.now().isoformat(),
        "run_id":      run_id,
        "script_name": script_name,
        "event":       event,
        **kwargs,
    }
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def init_run_summary() -> None:
    """Create run_summary.csv with header row if it does not exist."""
    if not RUN_SUMMARY_PATH.exists():
        with open(RUN_SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RUN_SUMMARY_COLUMNS).writeheader()


def write_run_summary(
    run_id:       str,
    model_name:   str,
    start_time:   str,
    end_time:     str   = "",
    best_epoch:   int   = -1,
    best_val_loss: float = float("inf"),
    test_loss:    float  = float("inf"),
    notes:        str   = "",
) -> None:
    """Append one row to run_summary.csv."""
    init_run_summary()
    row = {
        "run_id":       run_id,
        "model_name":   model_name,
        "start_time":   start_time,
        "end_time":     end_time,
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss":    test_loss,
        "notes":        notes,
    }
    with open(RUN_SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RUN_SUMMARY_COLUMNS).writerow(row)


def make_run_id(prefix: str = "run") -> str:
    """Generate a unique run_id from current timestamp."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
