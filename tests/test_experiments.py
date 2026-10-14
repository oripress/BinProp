"""Offline regression tests for the public experiment workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import OfflineClient

from binary_properties.provider import ProviderError
from binary_properties.types import Library, Position
from experiments.data import episode_from_dict, evenly_spaced, load_episodes
from experiments.prompts import direct_messages
from experiments.run import deliver, generate_direct, main
from experiments.runtime import evaluate_direct, helper_names
from experiments.scoring import hidden_score, rates, score_direct, select


def chess_row():
    def positions(offset, count):
        return [
            {"fen": f"4k3/8/8/8/8/8/4P3/4K3 w - - 0 {i}", "color": "white"}
            for i in range(offset, offset + count)
        ]

    return {
        "id": "example",
        "support": {"positive": positions(1, 4), "negative": positions(5, 4)},
        "held_out": {
            "positive": positions(9, 8),
            "negative": positions(17, 8),
            "nearby_negative": positions(25, 8),
        },
    }


class EpisodeTests(unittest.TestCase):
    def test_counts_and_player(self):
        row = chess_row()
        episode = episode_from_dict(row, "chess")
        self.assertEqual([len(v) for v in episode.hidden.values()], [8, 8, 8])
        row["held_out"]["positive"].pop()
        with self.assertRaises(ValueError):
            episode_from_dict(row, "chess")

    def test_disjoint_support(self):
        row = chess_row()
        row["held_out"]["negative"][0] = row["support"]["positive"][0]
        with self.assertRaises(ValueError):
            episode_from_dict(row, "chess")

    def test_even_support_indices(self):
        self.assertEqual(evenly_spaced(tuple(range(12))), (0, 4, 7, 11))

    def test_original_pool_format(self):
        row = chess_row()
        positives = row["support"]["positive"] + row["held_out"]["positive"]
        negatives = row["support"]["negative"] + row["held_out"]["negative"]
        stored = {
            "id": "example",
            "root_turn": "white",
            "positive_support_pool": positives,
            "negative_support_pool": negatives,
            "positive_holdout": positives,
            "normal_negative_holdout": negatives,
            "counterfactual_negative_holdout": row["held_out"]["nearby_negative"],
        }
        episode = episode_from_dict(stored, "chess")
        self.assertEqual(episode.positive[1].board, positives[4]["fen"])
        self.assertEqual(len(episode.hidden["positive"]), 8)

    def test_jsonl_error_is_located(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text('{"id": "broken"}\n')
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_episodes(path, "chess")

    def test_prompt_contains_only_support(self):
        episode = episode_from_dict(chess_row(), "chess")
        prompt = direct_messages(episode)[1]["content"]
        for p in (*episode.positive, *episode.negative):
            self.assertIn(p.board, prompt)
        for group in episode.hidden.values():
            for p in group:
                self.assertNotIn(p.board, prompt)
        self.assertIn(
            "Do not run commands",
            direct_messages(episode, no_tools=True)[1]["content"],
        )


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.episode = episode_from_dict(chess_row(), "chess")

    def test_minimum_and_average_two_motifs(self):
        candidates = [{"name": "a"}, {"name": "b"}]

        def evaluate(candidate, positions):
            self.assertEqual(len(positions), 24)
            n = 4 if candidate["name"] == "a" else 6
            return [True] * 8 + [False] * 8 + [False] * n + [True] * (8 - n)

        result = hidden_score(self.episode, candidates, evaluate)
        self.assertEqual(result["score"], 0.625)
        self.assertEqual(result["motif_count"], 2)
        self.assertEqual(hidden_score(self.episode, [], evaluate)["score"], 0)

    def test_invalid_is_not_false(self):
        self.assertEqual(
            rates([True] * 8 + [None] * 8 + [False] * 8, [8, 8, 8])["negative"],
            0,
        )

    def test_support_gate_and_fallback(self):
        rows = [
            {"name": "a", "support_fit": 0.25},
            {"name": "b", "support_fit": 0.0},
        ]
        self.assertEqual(select(rows), ([rows[0]], True))
        rows.append({"name": "c", "support_fit": 0.5})
        self.assertEqual(select(rows), ([rows[2]], False))

    def test_chess_invalid_hidden_helper_exclusion(self):
        source = "def explain_1(fen): return True\ndef explain_2(fen): return False"

        def evaluate(game, source, name, positions, **kwargs):
            if len(positions) == 8:
                return [True] * 4 + [False] * 4
            return [None] * 24 if name == "explain_1" else [True] * 8 + [False] * 16

        with patch("experiments.scoring.evaluate_direct", side_effect=evaluate):
            result = score_direct(self.episode, source)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["motif_count"], 1)
        self.assertTrue(result["errors"])

    def test_whole_fen_lookup_is_rejected(self):
        source = (
            'def explain_1(fen): return fen == "' + self.episode.positive[0].board + '"'
        )
        self.assertEqual(score_direct(self.episode, source)["score"], 0)


class RuntimeTests(unittest.TestCase):
    def test_helpers_not_aggregate_and_numeric_order(self):
        source = "def explain_3(fen): return True\ndef explain_1(fen): return False\ndef explain(fen): return True"
        self.assertEqual(helper_names(source, "chess"), ["explain_1", "explain_3"])

    def test_coordinate_specific_chess_is_allowed(self):
        p = episode_from_dict(chess_row(), "chess").positive[0]
        source = "import chess\ndef explain_1(fen):\n return chess.Board(fen).piece_at(chess.E2) is not None"
        self.assertEqual(evaluate_direct("chess", source, "explain_1", [p]), [True])

    def test_go_smoke_check_and_one_arg_compatibility(self):
        p = Position(tuple(["." * 19] * 19), "B")
        source = "def explain_1(board): return len(board) == 19"
        self.assertEqual(evaluate_direct("go", source, "explain_1", [p]), [True])
        with self.assertRaises(ValueError):
            evaluate_direct(
                "go",
                "def explain_1(board, color): return board[20] == color",
                "explain_1",
                [p],
            )

    def test_chess_invalid_outputs_remain_per_position(self):
        p = episode_from_dict(chess_row(), "chess").positive[0]
        for source in (
            "def explain_1(fen): return 1",
            "def explain_1(fen): return 1 / 0",
            "def explain_1(fen, color): return True",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    evaluate_direct("chess", source, "explain_1", [p]), [None]
                )

    def test_chess_file_constants_remain_available(self):
        p = episode_from_dict(chess_row(), "chess").positive[0]
        source = "import chess\ndef explain_1(fen): return chess.FILE_E == 4"
        self.assertEqual(evaluate_direct("chess", source, "explain_1", [p]), [True])

    def test_execution_deadline(self):
        p = episode_from_dict(chess_row(), "chess").positive[0]
        source = "def explain_1(fen):\n while True:\n  pass"
        with self.assertRaises(TimeoutError):
            evaluate_direct("chess", source, "explain_1", [p], timeout=0.2)


class RunnerTests(unittest.TestCase):
    def test_empty_response_retry_not_code_attempt(self):
        client = OfflineClient(
            [ProviderError("The model returned no text"), "visible answer"]
        )
        self.assertEqual(deliver(client, [], sleep=lambda _: None), "visible answer")
        self.assertEqual(len(client.messages), 2)

    def test_resume_reuses_delivered_response(self):
        client = OfflineClient(
            [AssertionError("Do not call a model for an already saved response")]
        )
        episode = episode_from_dict(chess_row(), "chess")
        previous = [{"attempt": 1, "response": "def explain_1(fen): return True"}]
        source, attempts = generate_direct(episode, client, previous_attempts=previous)
        self.assertEqual(source, previous[0]["response"])
        self.assertEqual(attempts, previous)

    def test_auth_failure_not_retried(self):
        client = OfflineClient(
            [ProviderError("Credentials rejected", retryable=False, status_code=401)]
        )
        with self.assertRaises(ProviderError):
            deliver(client, [], sleep=lambda _: None)
        self.assertEqual(len(client.messages), 1)

    def test_retry_decision_does_not_parse_error_message(self):
        for message, retryable in (
            ("Model endpoint returned HTTP 503", False),
            ("Set the API key; HTTP 401", True),
        ):
            failure = ProviderError(message, retryable=retryable)
            client = OfflineClient([failure, "answer"])
            sleep = Mock()
            with self.subTest(retryable=retryable):
                if retryable:
                    self.assertEqual(deliver(client, [], sleep=sleep), "answer")
                    sleep.assert_called_once_with(1)
                    self.assertEqual(len(client.messages), 2)
                else:
                    with self.assertRaises(ProviderError) as error:
                        deliver(client, [], sleep=sleep)
                    self.assertIs(error.exception, failure)
                    sleep.assert_not_called()
                    self.assertEqual(len(client.messages), 1)

    def test_provider_failures_stop_after_five_attempts(self):
        failure = ProviderError("Connection unavailable")
        client = OfflineClient([failure] * 5)
        sleep = Mock()
        with self.assertRaises(ProviderError) as error:
            deliver(client, [], sleep=sleep)
        self.assertIs(error.exception, failure)
        self.assertEqual(len(client.messages), 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4, 8])

    def test_invalid_code_is_delivered_attempt(self):
        client = OfflineClient(["not python"] * 3)
        episode = episode_from_dict(chess_row(), "chess")
        source, attempts = generate_direct(episode, client)
        self.assertIsNone(source)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(client.messages), 3)
        for attempt in attempts:
            for p in episode.hidden["nearby_negative"]:
                self.assertNotIn(p.board, attempt["messages"][1]["content"])

    def test_resume_skips_finished_cases_and_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "episodes.jsonl"
            data.write_text(json.dumps(chess_row()) + "\n")
            output = Path(directory) / "run"
            args = [
                "binprop",
                "--game",
                "chess",
                "--episodes",
                str(data),
                "--library",
                "chess/fable-5",
                "--output",
                str(output),
            ]
            with (
                patch(
                    "experiments.run.resolve_library",
                    return_value=Library("chess"),
                ),
                patch(
                    "experiments.run.score_binprop",
                    return_value={"id": "example", "score": 0.5},
                ) as score,
            ):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args + ["--resume"]), 0)
                self.assertEqual(score.call_count, 1)
                data.write_text(data.read_text() + "\n")
                self.assertEqual(main(args + ["--resume"]), 1)
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["mean_score"],
                0.5,
            )

    def test_new_run_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "episodes.jsonl"
            data.write_text(json.dumps(chess_row()) + "\n")
            output = Path(directory) / "existing"
            output.mkdir()
            with patch(
                "experiments.run.resolve_library", return_value=Library("chess")
            ):
                self.assertEqual(
                    main(
                        [
                            "binprop",
                            "--game",
                            "chess",
                            "--episodes",
                            str(data),
                            "--library",
                            "chess/fable-5",
                            "--output",
                            str(output),
                        ]
                    ),
                    1,
                )

    def test_direct_resume_skips_completed_case(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "episodes.jsonl"
            data.write_text(json.dumps(chess_row()) + "\n")
            output = Path(directory) / "run"
            args = [
                "direct",
                "--game",
                "chess",
                "--episodes",
                str(data),
                "--model",
                "test-model",
                "--output",
                str(output),
            ]
            with patch(
                "binary_properties.provider.ChatClient.complete",
                return_value="def explain_1(fen): return True",
            ) as complete:
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args + ["--resume"]), 0)
                self.assertEqual(complete.call_count, 1)
            row = json.loads((output / "cases/0000.json").read_text())
            self.assertEqual(row["motif_count"], 1)
            self.assertTrue(row["used_fallback"])

    def test_direct_resume_keeps_attempt_before_provider_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "episodes.jsonl"
            data.write_text(json.dumps(chess_row()) + "\n")
            output = Path(directory) / "run"
            args = [
                "direct",
                "--game",
                "chess",
                "--episodes",
                str(data),
                "--model",
                "test-model",
                "--output",
                str(output),
            ]
            with patch(
                "binary_properties.provider.ChatClient.complete",
                side_effect=[
                    "not python",
                    ProviderError(
                        "Credentials rejected", retryable=False, status_code=401
                    ),
                ],
            ) as complete:
                self.assertEqual(main(args), 1)
                self.assertEqual(complete.call_count, 2)
            attempts_path = output / "cases/0000-attempts.json"
            saved = json.loads(attempts_path.read_text())
            self.assertEqual(len(saved["attempts"]), 1)
            self.assertFalse((output / "cases/0000.json").exists())
            with patch(
                "binary_properties.provider.ChatClient.complete",
                return_value="def explain_1(fen): return True",
            ) as complete:
                self.assertEqual(main(args + ["--resume"]), 0)
                self.assertEqual(complete.call_count, 1)
            resumed = json.loads(attempts_path.read_text())
            self.assertEqual(resumed["attempts"][0], saved["attempts"][0])
            self.assertEqual([a["attempt"] for a in resumed["attempts"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
