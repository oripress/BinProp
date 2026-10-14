"""Do not attach a library's mismatched description to an executed predicate."""

import unittest

from binary_properties.inference import infer
from binary_properties.io import library_from_dict, library_to_dict
from binary_properties.types import Library, Motif, Position


def pawn_motif(name="More pawns"):
    description = "The evaluated player has more pawns."
    source = (
        "import chess\n"
        f"MOTIF_NAME = {name!r}\n"
        f"MOTIF_DESCRIPTION = {description!r}\n"
        "def motif(fen):\n"
        "    board = chess.Board(fen)\n"
        "    return len(board.pieces(chess.PAWN, board.turn)) > "
        "len(board.pieces(chess.PAWN, not board.turn))\n"
    )
    return Motif(name, description, source, 0.2)


class InferenceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.positive = [Position("7k/8/8/8/8/8/P7/K7 w - - 0 1", "white")]
        self.negative = [Position("7k/p7/8/8/8/8/8/K7 w - - 0 1", "white")]

    def test_serialized_name_or_description_mismatch_is_rejected(self):
        for field, text in (
            ("name", "Forced mate"),
            ("description", "The evaluated player can force checkmate."),
        ):
            with self.subTest(field=field):
                saved = library_to_dict(Library("chess", [pawn_motif()]))
                saved["motifs"][0][field] = text
                result = infer(library_from_dict(saved), self.positive, self.negative)
                self.assertEqual(result["selected"], [])
                self.assertEqual(result["ranked"], [])
                self.assertFalse(result["used_fallback"])
                self.assertEqual(len(result["errors"]), 1)
                self.assertEqual(result["errors"][0]["type"], "ValueError")
                self.assertIn("metadata does not match", result["errors"][0]["error"])

    def test_mismatch_does_not_exclude_other_valid_motifs(self):
        original = pawn_motif()
        bad = Motif("Forced mate", original.description, original.source, 1.0)
        good = pawn_motif("Valid pawn advantage")
        result = infer(Library("chess", [bad, good]), self.positive, self.negative)
        self.assertEqual([row["id"] for row in result["ranked"]], [good.id])
        self.assertEqual([row["id"] for row in result["selected"]], [good.id])
        self.assertEqual(result["selected"][0]["description"], good.description)
        self.assertEqual(result["selected"][0]["support_fit"], 1.0)
        self.assertEqual([row["id"] for row in result["errors"]], [bad.id])
        self.assertFalse(result["used_fallback"])

    def test_matching_serialized_metadata_is_preserved(self):
        original = pawn_motif()
        library = library_from_dict(library_to_dict(Library("chess", [original])))
        result = infer(library, self.positive, self.negative)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["selected"][0]["name"], original.name)
        self.assertEqual(result["selected"][0]["description"], original.description)
        self.assertEqual(result["selected"][0]["source"], original.source)
        self.assertEqual(result["selected"][0]["support_fit"], 1.0)


if __name__ == "__main__":
    unittest.main()
