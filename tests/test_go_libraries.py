"""Frozen Go library integrity and replay of one canonical real-game example."""

import importlib.util
import unittest
from hashlib import sha256
from pathlib import Path

from binary_properties.inference import infer
from binary_properties.io import position_from_dict
from binary_properties.libraries import list_libraries, load_builtin_library
from binary_properties.runtime import evaluate_source

EXPECTED_COUNTS = {
    "go/claude-opus-5": 35,
    "go/deepseek-v4-pro": 5,
    "go/fable-5": 45,
    "go/glm-5.2": 34,
    "go/gpt-5.5": 39,
    "go/gpt-5.6-sol": 31,
    "go/kimi-k3": 30,
    "go/qwen3.5-397b-a17b": 31,
}
ARCHIVE_SHA256 = "3578cc235b87ca4b08479b4ec87d3b6fae847e1bbdb3b1d8c7acb3c6c870342b"


def example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "explain_go.py"
    spec = importlib.util.spec_from_file_location("explain_go", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_hash(rows):
    return sha256("".join(rows).encode("ascii")).hexdigest()


class GoLibraryTests(unittest.TestCase):
    def test_complete_original_libraries_and_scores(self):
        catalog = [row for row in list_libraries() if row["game"] == "go"]
        self.assertEqual(
            {row["id"]: row["motif_count"] for row in catalog}, EXPECTED_COUNTS
        )
        self.assertEqual(sum(EXPECTED_COUNTS.values()), 250)
        for entry in catalog:
            with self.subTest(library=entry["id"]):
                library = load_builtin_library(entry["id"])
                self.assertEqual(library.game, "go")
                self.assertEqual(len(library.motifs), entry["motif_count"])
                self.assertEqual(library.metadata["library_id"], entry["id"])
                origin = library.metadata.get("parent_library", library.metadata)
                self.assertEqual(origin["archive_sha256"], ARCHIVE_SHA256)
                recorded = library.metadata["motif_provenance"]
                self.assertEqual(set(recorded), {m.id for m in library.motifs})
                self.assertEqual(
                    [recorded[m.id]["rank"] for m in library.motifs],
                    list(range(1, len(library.motifs) + 1)),
                )
                for motif in library.motifs:
                    provenance = recorded[motif.id]
                    self.assertEqual(
                        sha256(motif.source.encode()).hexdigest(),
                        provenance["source_sha256"],
                    )
                    self.assertAlmostEqual(
                        motif.delta,
                        provenance["p_true_winning"] - provenance["p_true_losing"],
                    )

    def test_real_sgf_replays_to_canonical_root_and_both_endpoints(self):
        example = example_module()
        self.assertEqual(sha256(example.SGF).hexdigest(), example.SGF_SHA256)
        root, _ = example.root_position()
        self.assertEqual(
            row_hash(example.board_rows(root)),
            "9b824368489140b4b11c0376b358c2aa7639344705c6376be73f26ec20bdfbcc",
        )
        groups = example.comparison()
        # Hashes from the original paper row, not generated from this example.
        expected = {
            "positive": "3122de1e84d93353a9b728cc595c44b63e46ccb8b560c52912ef7f664fdfd1ed",
            "negative": "7d130f549ad5a329401e0bb9ceddb8f836a1d09eacc8ed67742324953f20c6f2",
        }
        for group, digest in expected.items():
            self.assertEqual(row_hash(groups[group][0]["board_rows"]), digest)
            self.assertEqual(groups[group][0]["color"], "W")

    def test_every_original_motif_runs_on_real_game_for_both_colors(self):
        groups = example_module().comparison()
        positions = [
            position_from_dict({**rows[0], "color": color}, "go")
            for rows in groups.values()
            for color in ("B", "W")
        ]
        for identifier in EXPECTED_COUNTS:
            for motif in load_builtin_library(identifier).motifs:
                with self.subTest(library=identifier, motif=motif.name):
                    result = evaluate_source("go", motif.source, positions, timeout=10)
                    self.assertEqual(result.name, motif.name)
                    self.assertEqual(result.description, motif.description)
                    self.assertEqual(len(result.values), len(positions))
                    self.assertTrue(all(type(value) is bool for value in result.values))

    def test_real_game_inference_has_no_errors_or_fallback(self):
        example = example_module()
        groups = example.comparison()
        result = infer(
            load_builtin_library(example.LIBRARY_ID),
            [position_from_dict(row, "go") for row in groups["positive"]],
            [position_from_dict(row, "go") for row in groups["negative"]],
        )
        self.assertFalse(result["errors"])
        self.assertFalse(result["used_fallback"])
        self.assertTrue(result["selected"])
        self.assertTrue(all(row["support_fit"] == 1 for row in result["selected"]))


if __name__ == "__main__":
    unittest.main()
