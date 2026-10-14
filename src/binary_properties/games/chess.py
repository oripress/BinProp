"""Compile chess motifs and check player perspective and color symmetry."""

from __future__ import annotations

import ast

import chess

from .common import (
    CompiledMotif,
    MotifValidationError,
    argument_names,
    execute_source,
    parse_source,
    require_bool,
    require_color_usage,
)


def normalize_color(color: str) -> bool:
    if isinstance(color, str):
        if color.strip().lower() in {"w", "white"}:
            return chess.WHITE
        if color.strip().lower() in {"b", "black"}:
            return chess.BLACK
    raise MotifValidationError("chess color must be white or black")


def validate_board(fen: str) -> str:
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise MotifValidationError("chess board must be a complete six-field FEN")
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise MotifValidationError(f"invalid chess FEN: {exc}") from exc
    if not board.is_valid():
        raise MotifValidationError(
            f"invalid chess position (status={int(board.status())})"
        )
    return fen


_SYMMETRY_SMOKE_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "8/8/4p3/8/4P3/8/8/K6k w - - 0 1",
    "r3k2r/ppp2ppp/2n1bn2/3qp3/3QP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
)


def compile_motif(source: str) -> CompiledMotif:
    tree, fn = parse_source(source, {"chess", "math"})
    args = argument_names(fn)
    if fn.name == "motif":
        if len(args) != 1:
            raise MotifValidationError(
                "chess motif must define motif(fen: str) -> bool"
            )
        explicit_color = False
    else:
        if len(args) != 2:
            raise MotifValidationError(
                "compatible chess basis must define basis(fen, color) -> bool"
            )
        require_color_usage(fn)
        explicit_color = True
        if any(
            isinstance(node, ast.Attribute) and node.attr == "turn"
            for node in ast.walk(fn)
        ):
            raise MotifValidationError(
                "explicit-color chess basis must not use board.turn"
            )
    name, description, func = execute_source(tree, fn, {"chess", "math"}, chess=chess)

    def evaluate(fen: str, color: str) -> bool:
        side = normalize_color(color)
        if explicit_color:
            return require_bool(func(fen, side))
        # Evaluate for this color by changing the turn, not the pieces.
        board = chess.Board(fen)
        board.turn = side
        return require_bool(func(board.fen()))

    compiled = CompiledMotif(name, description, evaluate)
    validate_symmetry(compiled, _SYMMETRY_SMOKE_FENS)
    return compiled


def validate_symmetry(motif: CompiledMotif, fens) -> None:
    for fen in fens:
        mirrored = chess.Board(fen).mirror().fen()
        original_values = (motif.evaluate(fen, "white"), motif.evaluate(fen, "black"))
        mirrored_values = (
            motif.evaluate(mirrored, "white"),
            motif.evaluate(mirrored, "black"),
        )
        if mirrored_values != original_values[::-1]:
            raise MotifValidationError(
                "chess motif is not color-symmetric under board.mirror()"
            )
