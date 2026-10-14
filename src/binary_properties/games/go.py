"""Compile Go motifs and validate boards, colors, and Boolean outputs."""

from __future__ import annotations

from .common import (
    CompiledMotif,
    MotifValidationError,
    execute_source,
    parse_source,
    require_bool,
    require_color_usage,
)


def normalize_color(color: str) -> str:
    if isinstance(color, str):
        if color.strip().upper() in {"B", "BLACK"}:
            return "B"
        if color.strip().upper() in {"W", "WHITE"}:
            return "W"
    raise MotifValidationError("Go color must be B or W")


def validate_board(board_rows: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(board_rows, (list, tuple)) or len(board_rows) != 19:
        raise MotifValidationError("Go board must contain exactly 19 rows")
    if any(
        not isinstance(row, str)
        or len(row) != 19
        or any(stone not in ".BW" for stone in row)
        for row in board_rows
    ):
        raise MotifValidationError(
            "Go rows must be 19-character strings over '.', 'B', and 'W'"
        )
    return list(board_rows)


def compile_motif(source: str) -> CompiledMotif:
    tree, fn = parse_source(source, {"math", "sgfmill"})
    require_color_usage(fn)
    name, description, func = execute_source(tree, fn, {"math", "sgfmill"})

    def evaluate(board_rows: list[str] | tuple[str, ...], color: str) -> bool:
        return require_bool(func(validate_board(board_rows), normalize_color(color)))

    compiled = CompiledMotif(name, description, evaluate)
    # Komi can make color swaps asymmetric. Check Boolean outputs for both colors.
    empty = ["." * 19 for _ in range(19)]
    compiled.evaluate(empty, "B")
    compiled.evaluate(empty, "W")
    return compiled
