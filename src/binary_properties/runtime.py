"""Compile and evaluate motifs in a subprocess.

Code checks and filtered environment variables are not a security sandbox.
Isolate generated code from secrets, sensitive files, and network access.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Sequence

from .games.common import MotifValidationError, validate_motif_text
from .types import Evaluation, Game, Position
from .workers import run_worker


def evaluate_source(
    game: Game, source: str, positions: Sequence[Position], *, timeout: float = 60.0
) -> Evaluation:
    """Return strict booleans; timeout includes compilation and symmetry smoke calls.

    Invalid code, inputs, or return values raise MotifValidationError. An expired
    deadline raises TimeoutError. Reported timings cover supplied board calls only.
    """
    if game not in {"chess", "go"}:
        raise MotifValidationError(f"unsupported game: {game!r}")
    validate_motif_text(source, "source")
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    rows = []
    for position in positions:
        if not isinstance(position, Position):
            raise MotifValidationError("positions must contain Position records")
        if game == "chess" and not isinstance(position.board, str):
            raise MotifValidationError("chess positions require a FEN string")
        if game == "go" and not isinstance(position.board, (list, tuple)):
            raise MotifValidationError("Go positions require a sequence of board rows")
        rows.append({"board": position.board, "color": position.color})
    payload = json.dumps({"game": game, "source": source, "positions": rows})
    try:
        process = run_worker(
            "binary_properties.runtime", payload, args=("--worker",), timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"motif compilation/evaluation exceeded {timeout:g}s"
        ) from exc
    if process.returncode:
        raise MotifValidationError(
            f"motif worker exited with status {process.returncode}: {process.stderr.strip()[:1000]}"
        )
    try:
        result = json.loads(process.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MotifValidationError("motif worker returned an invalid response") from exc
    if result.get("error"):
        raise MotifValidationError(str(result["error"]))
    values = result.get("values")
    if (
        not isinstance(values, list)
        or len(values) != len(rows)
        or any(type(v) is not bool for v in values)
    ):
        raise MotifValidationError("motif worker returned invalid boolean outputs")
    return Evaluation(
        values,
        result["name"],
        result["description"],
        float(result["p95_ms"]),
        float(result["max_ms"]),
    )


def _evaluate_payload(payload: dict) -> Evaluation:
    game = payload["game"]
    if game == "chess":
        from .games import chess as adapter
    elif game == "go":
        from .games import go as adapter
    else:
        raise MotifValidationError(f"unsupported game: {game!r}")
    rows = payload["positions"]
    for row in rows:
        adapter.validate_board(row["board"])
        adapter.normalize_color(row["color"])
    motif = adapter.compile_motif(payload["source"])
    # Chess requires color symmetry; original Go motifs may account for komi.
    if game == "chess":
        adapter.validate_symmetry(motif, [row["board"] for row in rows[:512]])
    values, timings = [], []
    for row in rows:
        started = time.perf_counter()
        values.append(motif.evaluate(row["board"], row["color"]))
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    p95 = (
        timings[min(len(timings) - 1, int(round((len(timings) - 1) * 0.95)))]
        if timings
        else 0.0
    )
    return Evaluation(
        values, motif.name, motif.description, p95, max(timings, default=0.0)
    )


def _main() -> int:
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("runtime is internal; use the binary-properties command")
    try:
        result = asdict(_evaluate_payload(json.load(sys.stdin)))
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
