"""Regression tests for serialized metadata and game-specific color semantics."""

import tempfile
import unittest
from pathlib import Path

from binary_properties.discovery import DiscoveryConfig, discover
from binary_properties.io import load_library, save_library
from binary_properties.runtime import evaluate_source
from binary_properties.types import Library, Motif, Position


def pawn_source(name="More pawns", description="The evaluated side has more pawns."):
    return (
        "import chess\n"
        f"MOTIF_NAME = {name!r}\n"
        f"MOTIF_DESCRIPTION = {description!r}\n"
        "def motif(fen):\n"
        "    b = chess.Board(fen)\n"
        "    return len(b.pieces(chess.PAWN, b.turn)) > "
        "len(b.pieces(chess.PAWN, not b.turn))\n"
    )


POSITIONS = [
    Position("7k/8/8/8/8/8/P7/K7 w - - 0 1", "white", 1),
    Position("7k/p7/8/8/8/8/8/K7 w - - 0 1", "white", 0),
]


class OneProposal:
    def __init__(self, source):
        self.source = source

    def complete(self, messages):
        return chr(96) * 3 + "python\n" + self.source + chr(96) * 3


class MetadataRoundtripTests(unittest.TestCase):
    def test_maximum_metadata_survives_discovery_save_and_reload(self):
        source = pawn_source("n" * 180, "d" * 2000)
        library = discover(
            "chess",
            POSITIONS,
            OneProposal(source),
            config=DiscoveryConfig(iterations=1, max_p95_ms=100),
        )
        self.assertEqual(len(library.motifs), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.json"
            save_library(path, library)
            self.assertEqual(load_library(path), library)

    def test_oversized_metadata_is_rejected_before_acceptance(self):
        for name, description in (
            ("n" * 181, "Description"),
            ("Name", "d" * 2001),
        ):
            with self.subTest(
                name_length=len(name), description_length=len(description)
            ):
                events = []
                library = discover(
                    "chess",
                    POSITIONS,
                    OneProposal(pawn_source(name, description)),
                    config=DiscoveryConfig(iterations=1, max_p95_ms=100),
                    on_iteration=lambda event, current: events.append(event),
                )
                self.assertEqual(library.motifs, [])
                self.assertEqual(events[0]["reason"], "runtime_error")
                self.assertIn("Invalid motif", events[0]["error"])

    def test_invalid_metadata_cannot_overwrite_a_valid_checkpoint(self):
        good = Library("chess", [Motif("Name", "Description", pawn_source(), 1.0)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.json"
            save_library(path, good)
            original = path.read_bytes()
            bad = Library(
                "chess", [Motif("x" * 181, "Description", pawn_source(), 1.0)]
            )
            with self.assertRaisesRegex(ValueError, "Invalid motif name"):
                save_library(path, bad)
            self.assertEqual(path.read_bytes(), original)

    def test_blank_metadata_is_rejected_in_saved_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.json"
            for name, description in (("  ", "Description"), ("Name", "\n\t")):
                with self.subTest(name=name, description=description):
                    with self.assertRaises(ValueError):
                        save_library(
                            path,
                            Library(
                                "chess", [Motif(name, description, pawn_source(), 1.0)]
                            ),
                        )
                    self.assertFalse(path.exists())


def go_source(body):
    return (
        "MOTIF_NAME = 'Go predicate'\n"
        "MOTIF_DESCRIPTION = 'A synthetic predicate for validation.'\n"
        "def motif(board_rows, color):\n"
        f"    return {body}\n"
    )


class GoPerspectiveTests(unittest.TestCase):
    def test_komi_may_distinguish_colors_on_the_same_empty_board(self):
        source = go_source(
            "(sum(row.count(color) - row.count('W' if color == 'B' else 'B') "
            "for row in board_rows) + (6.5 if color == 'W' else -6.5)) > 0"
        )
        self.assertEqual(evaluate_source("go", source, []).values, [])
        empty = tuple(["." * 19] * 19)
        result = evaluate_source(
            "go", source, [Position(empty, "B"), Position(empty, "W")]
        )
        self.assertEqual(result.values, [False, True])

    def test_raw_board_and_evaluated_color_are_preserved(self):
        board = tuple(["." * 19] * 18 + ["." * 18 + "B"])
        source = go_source("board_rows[18][18] == 'B' and color == 'B'")
        result = evaluate_source(
            "go", source, [Position(board, "B"), Position(board, "W")]
        )
        self.assertEqual(result.values, [True, False])

    def test_color_relative_predicate_evaluates_both_players(self):
        black = tuple(["B" + "." * 18] + ["." * 19] * 18)
        white = tuple(["W" + "." * 18] + ["." * 19] * 18)
        source = go_source("board_rows[0][0] == color")
        result = evaluate_source(
            "go",
            source,
            [
                Position(black, "B"),
                Position(white, "W"),
                Position(black, "W"),
                Position(white, "B"),
            ],
        )
        self.assertEqual(result.values, [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
