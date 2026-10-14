"""Offline integration tests for run recovery and CLI continuation."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from helpers import CHESS_SOURCE, chess_row

from binary_properties.cli import main
from binary_properties.io import load_library
from binary_properties.provider import ProviderError
from binary_properties.runs import public_metadata, restore_outputs, run_lock


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.positions = self.root / "positions.jsonl"
        self.positions.write_text(
            json.dumps(chess_row(True, 1))
            + "\n"
            + json.dumps(chess_row(False, 0))
            + "\n"
        )
        self.run = self.root / "run"
        self.response = "```python\n" + CHESS_SOURCE + "```"

    def invoke(self, flags, responses=None, env=None):
        messages = []
        replies = iter(responses if responses is not None else [self.response] * 10)

        def complete(client, prompt):
            messages.append((client.model, client.base_url, deepcopy(prompt)))
            response = next(replies)
            if isinstance(response, Exception):
                raise response
            return response

        output, error = io.StringIO(), io.StringIO()
        with (
            patch.dict(
                os.environ, {"NEBIUS_API_KEY": "test-key", **(env or {})}, clear=True
            ),
            patch("binary_properties.provider.ChatClient.complete", complete),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            status = main(["discover", *flags])
        return status, messages, error.getvalue()

    def fresh(self, iterations=2):
        return self.invoke(
            [
                "--game",
                "chess",
                "--positions",
                str(self.positions),
                "--output",
                str(self.run),
                "--model",
                "original-model",
                "--max-p95-ms",
                "100",
                "--iterations",
                str(iterations),
            ]
        )

    def state(self):
        return json.loads((self.run / "checkpoint.json").read_text())

    def test_seed_builtin_by_id_preserves_paper_origin(self):
        code, messages, error = self.invoke(
            [
                "--game",
                "chess",
                "--positions",
                str(self.positions),
                "--output",
                str(self.run),
                "--model",
                "unused-model",
                "--initial",
                "chess/kimi-k3",
                "--iterations",
                "0",
            ]
        )
        self.assertEqual(code, 0, error)
        self.assertFalse(messages)
        library = load_library(self.run / "basis.json")
        self.assertEqual(len(library.motifs), 11)
        origin = library.metadata["initial_library"]
        self.assertEqual(origin["metadata"]["run_id"], "kimi3-chess-bp-fresh-20260921")
        self.assertEqual(len(origin["sha256"]), 64)

    def test_completed_outputs_match_checkpoint(self):
        for iterations in (0, 1):
            with self.subTest(iterations=iterations):
                self.run = self.root / f"run-{iterations}"
                code, calls, error = self.fresh(iterations=iterations)
                self.assertEqual(code, 0, error)
                self.assertEqual(len(calls), iterations)
                state = self.state()
                run = public_metadata(state["run"])
                self.assertEqual(json.loads((self.run / "run.json").read_text()), run)
                self.assertEqual(
                    load_library(self.run / "basis.json").metadata,
                    {**state["library"]["metadata"], **run},
                )

    def test_more_iterations_preserve_history_and_saved_provider(self):
        self.assertEqual(self.fresh()[0], 0)
        before = self.state()
        code, messages, error = self.invoke(
            ["--resume", str(self.run), "--iterations", "2"],
            env={
                "MOTIF_MODEL": "changed-env-model",
                "MOTIF_BASE_URL": "https://changed.example/v1",
            },
        )
        self.assertEqual(code, 0, error)
        self.assertEqual(
            messages[0][:2],
            ("original-model", "https://api.tokenfactory.nebius.com/v1"),
        )
        self.assertEqual(messages[0][2][1:-1], before["history"])
        self.assertEqual(json.loads(messages[0][2][-1]["content"])["iteration"], 2)
        after = self.state()
        self.assertEqual(after["next_iteration"], 4)
        self.assertEqual(after["event_count"], 4)
        self.assertEqual(after["config"]["iterations"], 4)
        self.assertEqual(
            len((self.run / "iterations.jsonl").read_text().splitlines()), 4
        )

    def test_provider_failure_retries_same_prompt_and_finishes_target(self):
        flags = [
            "--game",
            "chess",
            "--positions",
            str(self.positions),
            "--output",
            str(self.run),
            "--model",
            "original-model",
            "--max-p95-ms",
            "100",
            "--iterations",
            "3",
        ]
        code, calls, error = self.invoke(
            flags, [self.response, ProviderError("unavailable")]
        )
        self.assertEqual(code, 1)
        self.assertEqual(self.state()["next_iteration"], 1)
        code, retried, error = self.invoke(["--resume", str(self.run)])
        self.assertEqual(code, 0, error)
        self.assertEqual(retried[0][2], calls[-1][2])
        self.assertEqual(len(retried), 2)
        self.assertEqual(self.state()["next_iteration"], 3)
        self.assertEqual(self.state()["proposals"], 4)

    def test_changed_input_and_settings_rejected_without_output_changes(self):
        self.assertEqual(self.fresh()[0], 0)
        names = ["basis.json", "run.json", "checkpoint.json", "iterations.jsonl"]
        before = {name: (self.run / name).read_bytes() for name in names}
        code, calls, _ = self.invoke(
            ["--resume", str(self.run), "--model", "different"]
        )
        self.assertEqual(code, 1)
        self.assertFalse(calls)
        self.positions.write_text(self.positions.read_text() + "\n")
        code, calls, _ = self.invoke(["--resume", str(self.run), "--iterations", "1"])
        self.assertEqual(code, 1)
        self.assertFalse(calls)
        self.assertEqual(
            before, {name: (self.run / name).read_bytes() for name in names}
        )

    def test_incomplete_checkpoint_is_rejected_before_model_calls(self):
        self.assertEqual(self.fresh(iterations=0)[0], 0)
        state = self.state()
        del state["run"]["request_extra_sha256"]
        (self.run / "checkpoint.json").write_text(json.dumps(state))
        code, calls, error = self.invoke(["--resume", str(self.run)], responses=[])
        self.assertEqual(code, 1)
        self.assertFalse(calls)
        self.assertIn("Incomplete run checkpoint", error)
        self.assertEqual(self.state(), state)

    def test_byte_identical_relocated_input_is_allowed(self):
        self.assertEqual(self.fresh()[0], 0)
        moved = self.root / "moved.jsonl"
        self.positions.rename(moved)
        code, _, error = self.invoke(
            [
                "--resume",
                str(self.run),
                "--positions",
                str(moved),
                "--iterations",
                "1",
            ]
        )
        self.assertEqual(code, 0, error)
        self.assertEqual(self.state()["next_iteration"], 3)
        self.assertEqual(self.state()["run"]["positions_path"], str(moved))

    def test_recover_commit_before_derived_files_without_repeating_decision(self):
        def interrupted(directory, state):
            if state["event_count"] == 1:
                raise RuntimeError("simulated interruption after atomic commit")
            return restore_outputs(directory, state)

        with patch("binary_properties.runs.restore_outputs", side_effect=interrupted):
            code, calls, _ = self.fresh(iterations=3)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.state()["next_iteration"], 1)
        self.assertEqual((self.run / "iterations.jsonl").read_text(), "")
        code, calls, error = self.invoke(["--resume", str(self.run)])
        self.assertEqual(code, 0, error)
        self.assertEqual(len(calls), 2)
        events = [
            json.loads(line)
            for line in (self.run / "iterations.jsonl").read_text().splitlines()
        ]
        self.assertEqual([event["iteration"] for event in events], [0, 1, 2])
        self.assertEqual(len(load_library(self.run / "basis.json").motifs), 1)

    def test_run_lock_prevents_two_writers(self):
        self.assertEqual(self.fresh()[0], 0)
        with run_lock(self.run):
            code, calls, error = self.invoke(["--resume", str(self.run)])
        self.assertEqual(code, 1)
        self.assertIn("already active", error)
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
