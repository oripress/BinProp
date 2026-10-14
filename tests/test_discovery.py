"""Discovery tests use invented boards, offline clients, and tiny output vectors."""

import json
import unittest
from unittest.mock import patch

from helpers import OfflineClient, fenced, motif_source, synthetic_positions

from binary_properties.discovery import (
    DiscoveryConfig,
    activation_stats,
    discover,
    winning_containment,
)
from binary_properties.types import Evaluation, Library, Motif, Position


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.positions = synthetic_positions()

    def run_offline(
        self,
        vectors,
        *,
        game="chess",
        initial=(),
        names=None,
        config=None,
        positions=None,
        timings=None,
    ):
        names = names or list(vectors)
        sources = {name: motif_source(name) for name in vectors}
        source_names = {text: name for name, text in sources.items()}
        evaluations = []

        def evaluate(_game, text, rows, *, timeout):
            name = source_names[text]
            evaluations.append((name, _game, timeout, len(rows)))
            return Evaluation(
                list(vectors[name]),
                name,
                "Literal description",
                (timings or {}).get(name, 0.1),
                0.2,
            )

        library = Library(
            game,
            [
                Motif(name, "Literal description", sources[name], 999.0)
                for name in initial
            ],
        )
        client = OfflineClient([fenced(sources[name]) for name in names])
        events = []
        with patch("binary_properties.discovery.evaluate_source", side_effect=evaluate):
            result = discover(
                game,
                positions or self.positions,
                client,
                config=config or DiscoveryConfig(iterations=len(names)),
                initial_library=library,
                on_iteration=lambda event, state: events.append((event, state)),
            )
        return result, events, client, evaluations

    def test_delta_is_class_conditional_not_joint(self):
        stats = activation_stats([True, False, False, True], [1, 1, 1, 0])
        self.assertAlmostEqual(stats["delta"], 1 / 3 - 1)
        self.assertEqual(stats["activation_rate"], 0.5)
        with self.assertRaises(ValueError):
            activation_stats([True], [1])

    def test_overlap_uses_winning_rows_and_both_directions(self):
        self.assertEqual(
            winning_containment(
                [True, False, True, True], [True, True, False, False], [1, 1, 0, 0]
            ),
            1,
        )
        self.assertEqual(
            winning_containment([False, False, True], [True, False, True], [1, 1, 0]), 0
        )
        with self.assertRaises(ValueError):
            winning_containment([True], [], [1])

    def test_game_specific_delta_boundary(self):
        vectors = {"candidate": [True, True, False, False] + [False] * 4}
        for game, expected in (("chess", True), ("go", False)):
            with self.subTest(game=game):
                result, events, _, _ = self.run_offline(
                    vectors,
                    game=game,
                    config=DiscoveryConfig(iterations=1, min_delta=0.5),
                )
                self.assertEqual(events[0][0]["accepted"], expected)
                self.assertEqual(len(result.motifs), int(expected))

    def test_game_specific_containment_boundary(self):
        vectors = {
            "old": [True, True, False, False] + [False] * 4,
            "new": [False, True, True, False] + [False] * 4,
        }
        for game, expected in (("chess", False), ("go", True)):
            with self.subTest(game=game):
                _, events, _, _ = self.run_offline(
                    vectors,
                    game=game,
                    initial=("old",),
                    names=["new"],
                    config=DiscoveryConfig(iterations=1, max_containment=0.5),
                )
                self.assertEqual(events[0][0]["accepted"], expected)

    def test_replaces_every_weaker_overlapping_motif(self):
        vectors = {
            "left": [True, False, False, False] + [False] * 4,
            "right": [False, True, False, False] + [False] * 4,
            "combined": [True, True, True, False] + [False] * 4,
        }
        for game in ("chess", "go"):
            with self.subTest(game=game):
                result, events, _, _ = self.run_offline(
                    vectors, game=game, initial=("left", "right"), names=["combined"]
                )
                self.assertEqual([m.name for m in result.motifs], ["combined"])
                self.assertEqual(events[0][0]["removed"], ["left", "right"])
                self.assertEqual(result.motifs[0].delta, 0.75)
                self.assertEqual(events[0][1].motifs, result.motifs)

    def test_one_tied_blocker_prevents_component_replacement(self):
        vectors = {
            "weak": [True, False, False, False] + [False] * 4,
            "strong": [True, True, False, False] + [False] * 4,
            "candidate": [True, True, True, False, True, False, False, False],
        }
        result, events, _, _ = self.run_offline(
            vectors, initial=("weak", "strong"), names=["candidate"]
        )
        self.assertEqual([m.name for m in result.motifs], ["weak", "strong"])
        self.assertEqual(events[0][0]["reason"], "overlap_not_improved")
        self.assertEqual(events[0][0]["blocking_motif"], "strong")
        self.assertEqual(result.motifs[1].delta, 0.5)  # Stale saved 999 is recomputed.

    def test_cap_replaces_only_weakest_with_strict_improvement(self):
        vectors = {
            "old": [True, False, False, False] + [False] * 4,
            "tie": [False, True, False, False] + [False] * 4,
            "better": [False, True, True, False] + [False] * 4,
        }
        result, events, _, _ = self.run_offline(
            vectors,
            initial=("old",),
            names=["tie", "better"],
            config=DiscoveryConfig(iterations=2, max_motifs=1),
        )
        self.assertEqual(events[0][0]["reason"], "weakest_not_improved")
        self.assertEqual([m.name for m in result.motifs], ["better"])
        self.assertEqual(events[1][0]["removed"], ["old"])

    def test_cap_breaks_equal_delta_ties_by_library_order(self):
        vectors = {
            "first": [True, False, False, False] + [False] * 4,
            "second": [False, True, False, False] + [False] * 4,
            "better": [False, False, True, True] + [False] * 4,
        }
        for game in ("chess", "go"):
            with self.subTest(game=game):
                result, events, _, _ = self.run_offline(
                    vectors,
                    game=game,
                    initial=("first", "second"),
                    names=["better"],
                    config=DiscoveryConfig(iterations=1, max_motifs=2),
                )
                self.assertEqual([m.name for m in result.motifs], ["second", "better"])
                self.assertEqual(events[0][0]["removed"], ["first"])

    def test_duplicate_signature_precedes_overlap_blocker(self):
        vectors = {
            "same": [True, True, False, False] + [False] * 4,
            "blocker": [True, True, True, False] + [False] * 4,
            "candidate": [True, True, False, False] + [False] * 4,
        }
        result, events, _, _ = self.run_offline(
            vectors, initial=("same", "blocker"), names=["candidate"]
        )
        event = events[0][0]
        self.assertEqual(event["reason"], "duplicate_signature")
        self.assertEqual(event["duplicate_of"], "same")
        self.assertEqual(
            [row["name"] for row in event["overlaps"]], ["same", "blocker"]
        )
        self.assertNotIn("blocking_motif", event)
        self.assertEqual([m.name for m in result.motifs], ["same", "blocker"])

    def test_chess_activation_fraction_guard_is_not_applied_to_go(self):
        rows = [Position(str(i), "white", int(i < 200)) for i in range(400)]
        vectors = {"rare": [True] + [False] * 399}
        for game, accepted in (("chess", False), ("go", True)):
            with self.subTest(game=game):
                _, events, _, _ = self.run_offline(
                    vectors,
                    game=game,
                    positions=rows,
                    config=DiscoveryConfig(iterations=1, min_delta=0),
                )
                self.assertEqual(events[0][0]["accepted"], accepted)
                if not accepted:
                    self.assertEqual(
                        events[0][0]["reason"], "activation_rate_out_of_range"
                    )

    def test_game_specific_same_name_rule_uses_source_identity(self):
        old_source = motif_source("same name")
        improved_source = old_source + "# improved predicate version\n"
        initial = Library(
            "go", [Motif("same name", "Literal description", old_source, 0.25)]
        )

        def evaluate(game, text, rows, *, timeout):
            values = (
                [True, False, False, False]
                if text == old_source
                else [True, True, True, False]
            ) + [False] * 4
            return Evaluation(values, "same name", "Literal description", 0.1, 0.1)

        for game, accepted in (("chess", False), ("go", True)):
            initial.game = game
            events = []
            with (
                self.subTest(game=game),
                patch(
                    "binary_properties.discovery.evaluate_source", side_effect=evaluate
                ),
            ):
                result = discover(
                    game,
                    self.positions,
                    OfflineClient([fenced(improved_source)]),
                    config=DiscoveryConfig(iterations=1),
                    initial_library=initial,
                    on_iteration=lambda event, state: events.append(event),
                )
            self.assertEqual(events[0]["accepted"], accepted)
            if accepted:
                self.assertEqual(result.motifs[0].source, improved_source)
                self.assertEqual(events[0]["motif_id"], result.motifs[0].id)
                self.assertNotEqual(result.motifs[0].id, initial.motifs[0].id)
            else:
                self.assertEqual(events[0]["reason"], "duplicate_name")

    def test_speed_and_duplicate_gates(self):
        vectors = {"old": [True, False, False, False] + [False] * 4}
        _, events, _, _ = self.run_offline(vectors, timings={"old": 2.01})
        self.assertEqual(events[0][0]["reason"], "too_slow")
        vectors["new"] = vectors["old"]
        _, events, _, _ = self.run_offline(
            vectors,
            initial=("old",),
            names=["new"],
            config=DiscoveryConfig(iterations=1, max_containment=1),
        )
        self.assertEqual(events[0][0]["reason"], "duplicate_signature")

    def test_source_metadata_preserved_and_metadata_has_no_boards(self):
        name = "literal 'Title' – unchanged"
        result, events, _, _ = self.run_offline(
            {name: [True, False, False, False] + [False] * 4}
        )
        self.assertEqual(result.motifs[0].name, name)
        self.assertEqual(result.motifs[0].source, motif_source(name))
        self.assertEqual(result.motifs[0].description, "Literal description")
        self.assertNotIn("synthetic-board", json.dumps(result.metadata))
        self.assertNotIn("synthetic-board", json.dumps(events[0][0]))

    def test_concrete_feedback_and_bounded_history(self):
        vectors = {
            f"bad-{i}": [True, False, False, False, True, True, False, False]
            for i in range(8)
        }
        _, _, client, _ = self.run_offline(
            vectors, config=DiscoveryConfig(iterations=8, history_turns=2)
        )
        self.assertTrue(all(len(messages) <= 6 for messages in client.messages))
        feedback = client.messages[1][-2]["content"]
        self.assertIn("negative_activations", feedback)
        self.assertIn("synthetic-board-4", feedback)
        self.assertIn("positive_misses", feedback)

    def test_overlap_feedback_contains_shared_and_distinguishing_boards(self):
        vectors = {
            "old": [True, True, False, False] + [False] * 4,
            "new": [True, True, True, False, True, False, False, False],
            "later": [False, False, False, True] + [False] * 4,
        }
        _, _, client, _ = self.run_offline(
            vectors, initial=("old",), names=["new", "later"]
        )
        feedback = client.messages[1][-2]["content"]
        self.assertIn("winning_overlap", feedback)
        self.assertIn("winning_disagreement", feedback)
        self.assertIn("synthetic-board-2", feedback)

    def test_provider_error_checkpoints_then_propagates(self):
        rows = self.positions
        client = OfflineClient(
            [fenced(motif_source("good")), RuntimeError("offline unavailable")]
        )
        events = []
        with patch(
            "binary_properties.discovery.evaluate_source",
            return_value=Evaluation(
                [True, False, False, False] + [False] * 4,
                "good",
                "Literal description",
                0.1,
                0.1,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline unavailable"):
                discover(
                    "chess",
                    rows,
                    client,
                    config=DiscoveryConfig(iterations=100),
                    on_iteration=lambda event, state: events.append((event, state)),
                )
        self.assertEqual(len(client.messages), 2)
        self.assertEqual(events[-1][0]["reason"], "provider_error")
        self.assertEqual([m.name for m in events[-1][1].motifs], ["good"])
        self.assertEqual(events[-1][1].metadata["discovery"]["proposals"], 2)

    def test_malformed_response_and_runtime_failure_are_logged(self):
        client = OfflineClient(["no code", fenced(motif_source("bad"))])
        events = []
        with patch(
            "binary_properties.discovery.evaluate_source",
            side_effect=TimeoutError("deadline"),
        ):
            result = discover(
                "chess",
                self.positions,
                client,
                config=DiscoveryConfig(iterations=2),
                on_iteration=lambda event, state: events.append(event),
            )
        self.assertEqual(result.motifs, [])
        self.assertEqual(
            [e["reason"] for e in events], ["invalid_response", "runtime_error"]
        )
        self.assertIn("TimeoutError", events[1]["error"])

    def test_explicit_two_class_labels_required_before_provider_call(self):
        for labels in ([], [1], [0, None], [True, 0], [2, 0]):
            client = OfflineClient([])
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                discover(
                    "chess", [Position("x", "white", label) for label in labels], client
                )
            self.assertEqual(client.messages, [])

    def test_config_rejects_invalid_thresholds(self):
        for kwargs in (
            {"timeout": 0},
            {"max_p95_ms": float("nan")},
            {"min_delta": -1},
            {"max_containment": 0},
            {"iterations": True},
            {"history_turns": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DiscoveryConfig(**kwargs)

    def test_same_seed_produces_identical_prompts(self):
        vectors = {"one": [True, False, False, False] + [False] * 4}
        one = self.run_offline(vectors)[2]
        two = self.run_offline(vectors)[2]
        self.assertEqual(one.messages, two.messages)


class ActualRuntimeSmokeTests(unittest.TestCase):
    def test_chess_source_discovery_and_direct_execution_agree(self):
        text = (
            "import chess\n"
            "MOTIF_NAME = 'Pawn count advantage'\n"
            "MOTIF_DESCRIPTION = 'The evaluated player has more pawns.'\n"
            "def motif(fen: str) -> bool:\n"
            "    board = chess.Board(fen)\n"
            "    return len(board.pieces(chess.PAWN, board.turn)) > "
            "len(board.pieces(chess.PAWN, not board.turn))\n"
        )
        positions = [
            Position("7k/8/8/8/8/8/P7/K7 w - - 0 1", "white", 1),
            Position("7k/p7/8/8/8/8/8/K7 w - - 0 1", "white", 0),
        ]
        from binary_properties.runtime import evaluate_source

        direct = evaluate_source("chess", text, positions)
        self.assertEqual(direct.values, [True, False])
        result = discover(
            "chess",
            positions,
            OfflineClient([fenced(text)]),
            config=DiscoveryConfig(iterations=1, max_p95_ms=100),
        )
        self.assertEqual(len(result.motifs), 1)
        self.assertEqual(result.motifs[0].source, text)
        self.assertEqual(result.motifs[0].delta, 1)

    def test_go_source_discovery_and_direct_execution_agree(self):
        text = (
            "MOTIF_NAME = 'More stones'\n"
            "MOTIF_DESCRIPTION = 'The evaluated player has more stones.'\n"
            "def motif(board_rows: list[str], color: str) -> bool:\n"
            "    opponent = 'W' if color == 'B' else 'B'\n"
            "    return sum(row.count(color) for row in board_rows) > "
            "sum(row.count(opponent) for row in board_rows)\n"
        )
        black = ("B" + "." * 18,) + ("." * 19,) * 18
        white = ("W" + "." * 18,) + ("." * 19,) * 18
        positions = [Position(black, "B", 1), Position(white, "B", 0)]
        from binary_properties.runtime import evaluate_source

        direct = evaluate_source("go", text, positions)
        self.assertEqual(direct.values, [True, False])
        result = discover(
            "go",
            positions,
            OfflineClient([fenced(text)]),
            config=DiscoveryConfig(iterations=1, max_p95_ms=100),
        )
        self.assertEqual(len(result.motifs), 1)
        self.assertEqual(result.motifs[0].source, text)
        self.assertEqual(result.motifs[0].delta, 1)


if __name__ == "__main__":
    unittest.main()
