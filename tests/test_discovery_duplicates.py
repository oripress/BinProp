"""Duplicate proposals must remain safe to checkpoint and reload."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from binary_properties.discovery import DiscoveryConfig, discover
from binary_properties.io import load_library, save_library
from binary_properties.types import Library, Motif, Position

GO_SOURCE = (
    "MOTIF_NAME = 'Corner stone'\n"
    "MOTIF_DESCRIPTION = 'The evaluated side owns the top-left corner.'\n"
    "def motif(board_rows, color):\n"
    "    return board_rows[0][0] == color\n"
)
CHESS_SOURCE = (
    "import chess\n"
    "MOTIF_NAME = 'More pawns'\n"
    "MOTIF_DESCRIPTION = 'The evaluated side has more pawns.'\n"
    "def motif(fen):\n"
    "    board = chess.Board(fen)\n"
    "    return len(board.pieces(chess.PAWN, board.turn)) > "
    "len(board.pieces(chess.PAWN, not board.turn))\n"
)


def positions(game):
    if game == "go":
        return [
            Position(tuple([stone + "." * 18] + ["." * 19] * 18), "B", label)
            for stone, label in (("B", 1), ("W", 0))
        ]
    return [
        Position("7k/8/8/8/8/8/P7/K7 w - - 0 1", "white", 1),
        Position("7k/p7/8/8/8/8/8/K7 w - - 0 1", "white", 0),
    ]


def proposal_client(source):
    client = Mock()
    client.complete.return_value = chr(96) * 3 + "python\n" + source + chr(96) * 3
    return client


class DiscoveryDuplicateTests(unittest.TestCase):
    def test_go_duplicate_source_preserves_reloadable_checkpoints(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"

            def checkpoint(event, library):
                save_library(path, library)
                self.assertEqual(load_library(path), library)
                events.append(event)

            library = discover(
                "go",
                positions("go"),
                proposal_client(GO_SOURCE),
                config=DiscoveryConfig(iterations=2, max_containment=1, max_p95_ms=100),
                on_iteration=checkpoint,
            )
        self.assertEqual(len(library.motifs), 1)
        self.assertEqual(
            [event["reason"] for event in events],
            ["accepted_add", "duplicate_source"],
        )
        self.assertFalse(events[1]["accepted"])
        self.assertEqual(library.metadata["discovery"]["accepted_proposals"], 1)

    def test_chess_duplicate_name_keeps_precedence(self):
        events = []
        library = discover(
            "chess",
            positions("chess"),
            proposal_client(CHESS_SOURCE),
            config=DiscoveryConfig(iterations=2, max_containment=1, max_p95_ms=100),
            on_iteration=lambda event, library: events.append(event),
        )
        self.assertEqual(len(library.motifs), 1)
        self.assertEqual(
            [event["reason"] for event in events],
            ["accepted_add", "duplicate_name"],
        )

    def test_initial_duplicates_are_rejected_before_source_evaluation(self):
        for game, source in (("chess", CHESS_SOURCE), ("go", GO_SOURCE)):
            motif = Motif("Name", "Description", source, 1.0)
            client = Mock()
            with (
                self.subTest(game=game),
                patch("binary_properties.discovery.evaluate_source") as evaluate,
            ):
                with self.assertRaisesRegex(
                    ValueError, "Duplicate motif source in initial library"
                ):
                    discover(
                        game,
                        positions(game),
                        client,
                        initial_library=Library(game, [motif, motif]),
                        config=DiscoveryConfig(iterations=0),
                    )
                evaluate.assert_not_called()
                client.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
