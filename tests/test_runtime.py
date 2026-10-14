"""Synthetic contract tests; no model calls, datasets, or engines."""

import os
import subprocess
import time
import unittest
from unittest.mock import patch

import chess

from binary_properties.runtime import (
    MotifValidationError,
    evaluate_source,
)
from binary_properties.types import Position
from binary_properties.workers import worker_environment

EMPTY_GO = tuple("." * 19 for _ in range(19))
CHESS_FEN = "7k/8/8/8/8/8/1R6/K7 w - - 0 1"
CHESS_SOURCE = """
MOTIF_NAME = "own_rook"
MOTIF_DESCRIPTION = "The evaluated side has a rook."
import chess
def motif(fen: str) -> bool:
    board = chess.Board(fen)
    return bool(board.pieces(chess.ROOK, board.turn))
"""
GO_SOURCE = """
MOTIF_NAME = "own_stone"
MOTIF_DESCRIPTION = "The evaluated side has a stone on the board."
def motif(board_rows: list[str], color: str) -> bool:
    return any(color in row for row in board_rows)
"""


class RuntimeTests(unittest.TestCase):
    def test_chess_rewrites_turn_and_preserves_color_symmetry(self):
        fen_black_turn = CHESS_FEN.replace(" w ", " b ")
        mirrored = chess.Board(CHESS_FEN).mirror().fen()
        values = evaluate_source(
            "chess",
            CHESS_SOURCE,
            [
                Position(fen_black_turn, "white"),
                Position(CHESS_FEN, "black"),
                Position(mirrored, "black"),
                Position(mirrored, "white"),
            ],
        )
        self.assertEqual(values.values, [True, False, True, False])
        self.assertEqual(values.name, "own_rook")
        self.assertGreaterEqual(values.max_ms, values.p95_ms)
        self.assertGreaterEqual(values.p95_ms, 0.0)

    def test_go_uses_explicit_color_on_raw_board(self):
        rows = ("B" + "." * 18,) + EMPTY_GO[1:]
        result = evaluate_source(
            "go",
            GO_SOURCE,
            [
                Position(rows, "B"),
                Position(rows, "W"),
            ],
        )
        self.assertEqual(result.values, [True, False])

    def test_compatible_chess_explicit_color_basis(self):
        source = CHESS_SOURCE.replace(
            "def motif(fen: str)", "def basis(fen: str, color)"
        )
        source = source.replace("board.turn", "color")
        result = evaluate_source(
            "chess",
            source,
            [
                Position(CHESS_FEN, "white"),
                Position(CHESS_FEN, "black"),
            ],
        )
        self.assertEqual(result.values, [True, False])

    def test_go_sgfmill_import_remains_compatible(self):
        source = GO_SOURCE.replace("def motif", "from sgfmill import boards\ndef motif")
        self.assertEqual(
            evaluate_source("go", source, [Position(EMPTY_GO, "B")]).values, [False]
        )

    def test_chess_asymmetric_motif_rejected(self):
        source = CHESS_SOURCE.replace("board.turn", "chess.WHITE")
        with self.assertRaisesRegex(MotifValidationError, "color-symmetric"):
            evaluate_source("chess", source, [Position(CHESS_FEN, "white")])

    def test_strict_boolean_returns(self):
        cases = [
            (
                "chess",
                CHESS_SOURCE.replace("bool(board.pieces(chess.ROOK, board.turn))", "1"),
                Position(CHESS_FEN, "white"),
            ),
            (
                "go",
                GO_SOURCE.replace(
                    "any(color in row for row in board_rows)", "int(color == 'B')"
                ),
                Position(EMPTY_GO, "B"),
            ),
        ]
        for game, source, position in cases:
            with (
                self.subTest(game=game),
                self.assertRaisesRegex(MotifValidationError, "expected bool"),
            ):
                evaluate_source(game, source, [position])

    def test_missing_metadata_rejected(self):
        for field in ("MOTIF_NAME", "MOTIF_DESCRIPTION"):
            source = "\n".join(
                line for line in GO_SOURCE.splitlines() if not line.startswith(field)
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(MotifValidationError, field),
            ):
                evaluate_source("go", source, [Position(EMPTY_GO, "B")])

    def test_unknown_game_and_mismatched_board_types(self):
        cases = [
            ("shogi", GO_SOURCE, Position(EMPTY_GO, "B")),
            ("chess", CHESS_SOURCE, Position(EMPTY_GO, "white")),
            ("go", GO_SOURCE, Position(CHESS_FEN, "B")),
        ]
        for game, source, position in cases:
            with self.subTest(game=game), self.assertRaises(MotifValidationError):
                evaluate_source(game, source, [position])

    def test_invalid_boards_and_colors(self):
        cases = [
            ("chess", CHESS_SOURCE, Position("invalid", "white")),
            ("chess", CHESS_SOURCE, Position("8/8/8/8/8/8/8/8 w - - 0 1", "white")),
            ("chess", CHESS_SOURCE, Position(CHESS_FEN, "green")),
            ("go", GO_SOURCE, Position(EMPTY_GO[:-1], "B")),
            ("go", GO_SOURCE, Position(("X" * 19,) + EMPTY_GO[1:], "B")),
            ("go", GO_SOURCE, Position(EMPTY_GO, "green")),
        ]
        for game, source, position in cases:
            with (
                self.subTest(game=game, position=position),
                self.assertRaises(MotifValidationError),
            ):
                evaluate_source(game, source, [position])

    def test_forbidden_imports(self):
        for game, source, position in [
            ("chess", CHESS_SOURCE, Position(CHESS_FEN, "white")),
            ("go", GO_SOURCE, Position(EMPTY_GO, "B")),
        ]:
            with (
                self.subTest(game=game),
                self.assertRaisesRegex(MotifValidationError, "not allowed"),
            ):
                evaluate_source(game, "import os\n" + source, [position])

    def test_top_level_infinite_loop_deadline(self):
        started = time.monotonic()
        children = []
        popen = subprocess.Popen

        def capture(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            return child

        with patch("binary_properties.workers.subprocess.Popen", side_effect=capture):
            with self.assertRaises(TimeoutError):
                evaluate_source(
                    "go",
                    GO_SOURCE + "\nwhile True:\n    pass\n",
                    [Position(EMPTY_GO, "B")],
                    timeout=0.3,
                )
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        self.assertNotEqual(children[0].returncode, 0)

    def test_supplied_board_infinite_loop_deadline(self):
        source = GO_SOURCE.replace(
            "    return any(color in row for row in board_rows)",
            "    while board_rows[0][0] == color:\n        pass\n    return False",
        )
        # Empty smoke boards terminate; the supplied board triggers the loop.
        rows = ("B" + "." * 18,) + EMPTY_GO[1:]
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            evaluate_source("go", source, [Position(rows, "B")], timeout=0.3)
        self.assertLess(time.monotonic() - started, 3)

    def test_smoke_call_infinite_loop_deadline(self):
        source = CHESS_SOURCE.replace(
            "    board = chess.Board(fen)",
            "    while True:\n        pass\n    board = chess.Board(fen)",
        )
        with self.assertRaises(TimeoutError):
            evaluate_source(
                "chess", source, [Position(CHESS_FEN, "white")], timeout=0.3
            )

    def test_api_credentials_not_forwarded_to_worker(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "synthetic-secret",
                "UNUSUAL_PROVIDER_CREDENTIAL": "synthetic-secret",
                "PYTHONSTARTUP": "/synthetic/startup.py",
                "PYTHONPATH": "/synthetic/imports",
            },
        ):
            environment = worker_environment()
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("UNUSUAL_PROVIDER_CREDENTIAL", environment)
        self.assertNotIn("PYTHONSTARTUP", environment)
        self.assertNotIn("/synthetic/imports", environment["PYTHONPATH"])

    def test_empty_position_batch_still_compiles(self):
        result = evaluate_source("go", GO_SOURCE, [])
        self.assertEqual(result.values, [])
        self.assertEqual((result.p95_ms, result.max_ms), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
