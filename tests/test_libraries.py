"""Canonical library integrity and offline execution on synthetic chess boards."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from pathlib import Path

import chess

from binary_properties.cli import main
from binary_properties.io import save_library
from binary_properties.libraries import (
    list_libraries,
    load_builtin_library,
    resolve_library,
)
from binary_properties.runtime import evaluate_source
from binary_properties.types import Position

EXPECTED_COUNTS = {
    "chess/claude-opus-5": 30,
    "chess/deepseek-v4-pro": 28,
    "chess/fable-5": 20,
    "chess/glm-5.2": 21,
    "chess/gpt-5.5": 44,
    "chess/gpt-5.6-sol": 21,
    "chess/kimi-k3": 11,
    "chess/qwen3.5-397b-a17b": 17,
}


def synthetic_positions():
    fens = [
        chess.STARTING_FEN,
        "7k/8/8/8/8/8/P7/K7 w - - 0 1",
        "7k/7p/8/8/8/8/8/K7 w - - 0 1",
        "7k/8/8/8/8/8/8/K7 w - - 0 1",
    ]
    for line in (
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7",
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O",
        "e4 c5 e5 d5 exd6",
    ):
        board = chess.Board()
        for move in line.split():
            board.push_san(move)
        fens.append(board.fen())
    return [Position(fen, color) for fen in fens for color in ("white", "black")]


class LibraryTests(unittest.TestCase):
    def test_complete_catalog_and_original_source_hashes(self):
        catalog = [entry for entry in list_libraries() if entry["game"] == "chess"]
        self.assertEqual(
            {row["id"]: row["motif_count"] for row in catalog}, EXPECTED_COUNTS
        )
        for entry in catalog:
            with self.subTest(library=entry["id"]):
                library = load_builtin_library(entry["id"])
                self.assertEqual(library.game, "chess")
                self.assertEqual(len(library.motifs), entry["motif_count"])
                self.assertEqual(library.metadata["library_id"], entry["id"])
                metadata = library.metadata["motif_provenance"]
                self.assertEqual(set(metadata), {m.id for m in library.motifs})
                self.assertEqual(
                    [metadata[m.id]["rank"] for m in library.motifs],
                    list(range(1, len(library.motifs) + 1)),
                )
                dataset = library.metadata["dataset"]
                self.assertEqual(
                    dataset["sha256"],
                    "6bedf361f6189bcbeafab33fc38ab45cde0c5fbfb8a82e04beb8070c4ab972fa",
                )
                for motif in library.motifs:
                    recorded = metadata[motif.id]
                    self.assertEqual(
                        sha256(motif.source.encode("utf-8")).hexdigest(),
                        recorded["source_sha256"],
                    )
                    self.assertAlmostEqual(
                        motif.delta,
                        recorded["winning_true"] / dataset["winning_count"]
                        - recorded["losing_true"] / dataset["losing_count"],
                    )

    def test_canonical_kimi_is_complete_fresh_paper_run(self):
        library = load_builtin_library("chess/kimi-k3")
        self.assertEqual(library.metadata["run_id"], "kimi3-chess-bp-fresh-20260921")
        self.assertEqual(library.metadata["model_id"], "moonshotai/Kimi-K3")
        self.assertEqual(len(library.motifs), 11)
        self.assertEqual(library.motifs[0].name, "material edge with activity tiebreak")
        self.assertAlmostEqual(library.motifs[0].delta, 0.65221)

    def test_every_original_motif_runs_in_bounded_worker(self):
        positions = synthetic_positions()
        for entry in list_libraries():
            if entry["game"] != "chess":
                continue
            for motif in load_builtin_library(entry["id"]).motifs:
                with self.subTest(library=entry["id"], motif=motif.name):
                    result = evaluate_source(
                        "chess", motif.source, positions, timeout=10
                    )
                    self.assertEqual(result.name, motif.name)
                    self.assertEqual(result.description, motif.description)
                    self.assertEqual(len(result.values), len(positions))
                    self.assertTrue(all(type(value) is bool for value in result.values))

    def test_file_library_roundtrip_and_unknown_names(self):
        library = load_builtin_library("chess/kimi-k3")
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "my-library.json"
            save_library(file, library)
            self.assertEqual(resolve_library(file), library)
            self.assertEqual(resolve_library(str(file)), library)
        self.assertEqual(resolve_library("chess/kimi-k3"), library)
        for name in ("chess/unknown", "../catalog", "chess/kimi27"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_builtin_library(name)

    def test_cli_lists_and_uses_builtins_without_credentials(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["libraries", "--json"]), 0)
        self.assertEqual(len(json.loads(stdout.getvalue())), 16)
        with tempfile.TemporaryDirectory() as directory:
            positions = Path(directory) / "compare.json"
            positions.write_text(
                json.dumps(
                    {
                        "positive": [{"fen": "7k/8/8/8/8/8/P7/K7 w - - 0 1"}],
                        "negative": [{"fen": "7k/7p/8/8/8/8/8/K7 w - - 0 1"}],
                    }
                )
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "infer",
                        "--library",
                        "chess/kimi-k3",
                        "--positions",
                        str(positions),
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["ranked"])
            self.assertFalse(result["errors"])


if __name__ == "__main__":
    unittest.main()
