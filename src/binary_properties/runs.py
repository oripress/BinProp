"""Save discovery checkpoints and recover interrupted writes."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from .io import library_from_dict, save_library, write_json


@contextmanager
def run_lock(directory: Path, *, create: bool = False):
    """One writer per run; the OS releases the lock after a crash."""
    try:
        import fcntl
    except ImportError:
        raise ValueError("Discovery run locking requires Linux or macOS") from None
    if create:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This discovery run is already active") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def public_metadata(run: dict) -> dict:
    # Resume needs these private details; exported libraries do not.
    return {
        key: value
        for key, value in run.items()
        if key not in ("positions_path", "request_extra")
    }


def restore_outputs(directory: Path, state: dict) -> None:
    """Repair derived files if a process stopped just after checkpoint commit."""
    count = state["event_count"]
    last = state.get("last_event")
    log = directory / "iterations.jsonl"
    current = log.read_text(encoding="utf-8") if log.exists() else ""
    lines = current.splitlines()
    events = []
    for i, text in enumerate(lines):
        try:
            events.append(json.loads(text))
        except ValueError:
            # A crash can leave the newest committed row partially appended.
            if i != count - 1 or i != len(lines) - 1:
                raise ValueError(
                    "Iteration log is inconsistent with the checkpoint"
                ) from None
    if len(events) == count - 1 and isinstance(last, dict):
        events.append(last)
    elif len(events) != count or (count and events[-1] != last):
        raise ValueError("Iteration log is inconsistent with the checkpoint")
    expected = "".join(
        json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        for event in events
    )
    if expected != current or not log.exists():
        # The checkpoint holds the latest row, so resume can retry this write.
        temp = directory / ".iterations.jsonl.tmp"
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, log)
    library = library_from_dict(state["library"])
    library.metadata.update(public_metadata(state["run"]))
    save_library(directory / "basis.json", library)
    write_json(directory / "run.json", public_metadata(state["run"]))


def save_checkpoint(directory: Path, state: dict, run: dict) -> None:
    run = {**run, "config": state["config"]}
    committed = {**state, "run": run}
    # All other files can be rebuilt from this commit and the existing journal.
    write_json(directory / "checkpoint.json", committed)
    restore_outputs(directory, committed)
