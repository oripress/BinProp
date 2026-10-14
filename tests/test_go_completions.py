"""Pin the final Go completions independently of generated catalog checksums."""

import unittest

from binary_properties.libraries import load_builtin_library

COMPLETIONS = {
    "go/claude-opus-5": {
        "model": "opus5",
        "count": 35,
        "parent_count": 33,
        "additional_iterations": 10,
        "summary_sha256": "95d8f1da970d8b999c61ecb239b61540aebd6984dcb454bba0e29f1499892660",
        "ranking_sha256": "4944a50589bed5feef52c19e9993d94cf8280951a1a80c840307e2380c288a14",
        "new_names": {
            "level on the count but ahead on doomed material",
            "untouched home zones",
        },
    },
    "go/kimi-k3": {
        "model": "kimi3",
        "count": 30,
        "parent_count": 29,
        "additional_iterations": 4,
        "summary_sha256": "58dfd0e2feec1cc9000672b44d155b936f094fb454871033d0a45bb6f6accd90",
        "ranking_sha256": "3727db3e4eb299bfafa900944d6d797b93ebfb0d87aa357723fb6033f42fedfe",
        "new_names": {"enclosed swing area advantage"},
    },
}


class GoCompletionTests(unittest.TestCase):
    def test_final_completions_and_parent_provenance(self):
        for identifier, expected in COMPLETIONS.items():
            with self.subTest(library=identifier):
                library = load_builtin_library(identifier)
                metadata = library.metadata
                self.assertEqual(len(library.motifs), expected["count"])
                self.assertEqual(metadata["completion_status"], "complete")
                self.assertEqual(metadata["successful_iterations"], 100)
                self.assertEqual(
                    metadata["run_id"],
                    "go-bp-completion-20260924/completed/runs/" + expected["model"],
                )
                self.assertEqual(
                    metadata["additional_iterations"], expected["additional_iterations"]
                )
                self.assertEqual(
                    metadata["completion_summary_sha256"], expected["summary_sha256"]
                )
                self.assertEqual(metadata["ranking_sha256"], expected["ranking_sha256"])
                self.assertEqual(
                    metadata["parent_library"]["motif_count"], expected["parent_count"]
                )
                self.assertNotIn("archive_sha256", metadata)
                self.assertTrue(
                    expected["new_names"] <= {m.name for m in library.motifs}
                )


if __name__ == "__main__":
    unittest.main()
