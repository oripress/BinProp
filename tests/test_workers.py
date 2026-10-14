"""Shared worker transport tests; no generated code or network requests."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from binary_properties.workers import run_worker, worker_environment


class WorkerTests(unittest.TestCase):
    def test_launch_preserves_command_streams_and_remaining_deadline(self):
        process = Mock(returncode=7)
        process.communicate.return_value = ("output", "error")
        with (
            patch(
                "binary_properties.workers.subprocess.Popen", return_value=process
            ) as launch,
            patch("binary_properties.workers.time.monotonic", side_effect=[10, 10.25]),
        ):
            result = run_worker(
                "example.worker", "payload", args=("--worker",), timeout=1
            )
        command = [sys.executable, "-m", "example.worker", "--worker"]
        self.assertEqual(launch.call_args.args, (command,))
        self.assertEqual(result.args, command)
        self.assertEqual(
            (result.returncode, result.stdout, result.stderr), (7, "output", "error")
        )
        process.communicate.assert_called_once_with("payload", timeout=0.75)
        process.kill.assert_not_called()
        self.assertEqual(launch.call_args.kwargs["encoding"], "utf-8")
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(launch.call_args.kwargs[stream], subprocess.PIPE)

    def test_timeout_and_interrupt_kill_and_reap_worker(self):
        for failure in (
            subprocess.TimeoutExpired("worker", 1),
            KeyboardInterrupt(),
            BrokenPipeError("pipe closed"),
        ):
            process = Mock(returncode=None)
            process.communicate.side_effect = [failure, ("", "")]
            with (
                self.subTest(failure=type(failure)),
                patch(
                    "binary_properties.workers.subprocess.Popen", return_value=process
                ),
                self.assertRaises(type(failure)) as error,
            ):
                run_worker("example.worker", "payload", timeout=1)
            self.assertIs(error.exception, failure)
            process.kill.assert_called_once_with()
            self.assertEqual(process.communicate.call_count, 2)
            self.assertEqual(process.communicate.call_args.args, ())

    def test_environment_allows_only_worker_settings(self):
        with patch.dict(
            os.environ,
            {"API_KEY": "private", "PYTHONPATH": "/untrusted", "LANG": "C.UTF-8"},
            clear=True,
        ):
            env = worker_environment(extra_path=Path("/research"))
        self.assertEqual(
            set(env),
            {"LANG", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"},
        )
        self.assertEqual(env["LANG"], "C.UTF-8")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], "/research")
        self.assertNotIn("/untrusted", env["PYTHONPATH"])
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
