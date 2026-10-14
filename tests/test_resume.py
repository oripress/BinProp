"""Resume regressions use deterministic offline clients and invented boards."""

import json
import unittest
from dataclasses import replace
from hashlib import sha256
from unittest.mock import patch

from helpers import OfflineClient, fenced, motif_source, synthetic_positions

from binary_properties.discovery import DiscoveryConfig, discover
from binary_properties.io import library_to_dict
from binary_properties.types import Evaluation, Library, Motif


def serialized(value):
    return json.loads(json.dumps(value, allow_nan=False))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.positions = synthetic_positions("private-resume-board")
        self.vectors = {
            "left": [True, False, False, False] + [False] * 4,
            "right": [False, True, False, False] + [False] * 4,
            "bad": [True, False, False, False, True, True, False, False],
            "merged": [True, True, True, False] + [False] * 4,
            "tail": [False, False, False, True] + [False] * 4,
            "duplicate": [True, True, True, False] + [False] * 4,
            "all": [True] * 4 + [False] * 4,
        }
        self.sources = {
            name: motif_source(name, description="Synthetic resume predicate")
            for name in self.vectors
        }
        self.source_names = {text: name for name, text in self.sources.items()}
        self.initial = Library(
            "chess",
            [
                Motif(name, "Synthetic resume predicate", self.sources[name], 0.1)
                for name in ("left", "right")
            ],
            {"origin": {"dataset": "original-pool", "authors": ["Original author"]}},
        )
        self.config = DiscoveryConfig(iterations=1, seed=71, history_turns=2)
        self.evaluator = patch(
            "binary_properties.discovery.evaluate_source", side_effect=self.evaluate
        ).start()
        self.addCleanup(patch.stopall)

    def evaluate(self, game, text, positions, *, timeout):
        name = self.source_names[text]
        return Evaluation(
            list(self.vectors[name]), name, "Synthetic resume predicate", 0.1, 0.2
        )

    def responses(self, names):
        return [fenced(self.sources[name]) for name in names]

    def run_segment(self, names, *, resume_state=None, game="chess"):
        client = OfflineClient(self.responses(names))
        events, states = [], []
        initial = self.initial if resume_state is None else None
        if initial is not None:
            initial = Library(game, initial.motifs, initial.metadata)
        result = discover(
            game,
            self.positions,
            client,
            config=replace(self.config, iterations=len(names)),
            initial_library=initial,
            resume_state=resume_state,
            on_iteration=lambda event, current: events.append(serialized(event)),
            on_checkpoint=lambda state: states.append(serialized(state)),
        )
        return result, client.messages, events, states

    def test_segmented_run_preserves_prompts_decisions_basis_and_provenance(self):
        names = ["bad", "merged", "tail", "duplicate", "all"]
        for game in ("chess", "go"):
            with self.subTest(game=game):
                full, messages, events, _ = self.run_segment(names, game=game)
                state = None
                resumed_messages, resumed_events = [], []
                for segment in (names[:1], names[1:3], names[3:]):
                    result, sent, decided, states = self.run_segment(
                        segment, resume_state=state, game=game
                    )
                    state = states[-1]
                    resumed_messages.extend(sent)
                    resumed_events.extend(decided)
                self.assertEqual(resumed_messages, messages)
                self.assertEqual(resumed_events, events)
                self.assertEqual(library_to_dict(result), library_to_dict(full))
                self.assertEqual([motif.name for motif in result.motifs], ["all"])
                self.assertEqual(state["next_iteration"], 5)
                self.assertEqual(state["proposals"], 5)
                self.assertEqual(state["accepted_proposals"], 3)
                self.assertEqual(state["event_count"], 5)
                self.assertEqual(state["initial_motifs"], 2)
                self.assertEqual(state["last_event"], events[-1])
                self.assertEqual(len(state["history"]), 4)
                discovery = result.metadata["discovery"]
                self.assertEqual(discovery["config"]["iterations"], 5)
                self.assertEqual(discovery["initial_motifs"], 2)
                self.assertEqual(discovery["proposals"], 5)
                self.assertEqual(discovery["accepted_proposals"], 3)
                provenance = result.metadata["initial_library"]
                self.assertEqual(provenance["metadata"], self.initial.metadata)
                self.assertEqual(len(provenance["sha256"]), 64)
                self.assertNotIn("private-resume-board", json.dumps(result.metadata))
                self.assertNotIn("private-resume-board", json.dumps(events))
                self.assertIn("private-resume-board", json.dumps(state["history"]))

    def test_provider_failure_retries_same_prompt_and_keeps_attempt_counts(self):
        names = ["bad", "merged", "tail"]
        expected, messages, events, _ = self.run_segment(names)
        client = OfflineClient(
            self.responses(["bad"]) + [RuntimeError("offline unavailable")]
        )
        failed_events, states = [], []
        with self.assertRaisesRegex(RuntimeError, "offline unavailable"):
            discover(
                "chess",
                self.positions,
                client,
                config=replace(self.config, iterations=3),
                initial_library=self.initial,
                on_iteration=lambda event, current: failed_events.append(
                    serialized(event)
                ),
                on_checkpoint=lambda state: states.append(serialized(state)),
            )
        before_failure, failed = states[-2:]
        self.assertEqual(failed["next_iteration"], 1)
        self.assertEqual(failed["proposals"], 2)
        self.assertEqual(failed["event_count"], 2)
        self.assertEqual(failed["history"], before_failure["history"])
        self.assertEqual(
            failed["library"]["motifs"], before_failure["library"]["motifs"]
        )
        self.assertEqual(failed["last_event"]["reason"], "provider_error")
        result, retried, decisions, resumed = self.run_segment(
            names[1:], resume_state=failed
        )
        self.assertEqual(client.messages, messages[:2])
        self.assertEqual(retried, messages[1:])
        self.assertEqual(retried[0], client.messages[-1])
        self.assertEqual(failed_events[:1] + decisions, events)
        self.assertEqual(result.motifs, expected.motifs)
        self.assertEqual(
            result.metadata["initial_library"], expected.metadata["initial_library"]
        )
        self.assertEqual(result.metadata["discovery"]["proposals"], 4)
        self.assertEqual(result.metadata["discovery"]["accepted_proposals"], 2)
        self.assertEqual(result.metadata["discovery"]["config"]["iterations"], 3)
        self.assertEqual(resumed[-1]["next_iteration"], 3)
        self.assertEqual(resumed[-1]["event_count"], 4)

    def test_checkpoint_precedes_requests_and_iteration_notifications(self):
        calls, states = [], []
        client = OfflineClient(self.responses(["bad"]), calls)

        def checkpoint(state):
            calls.append("checkpoint")
            states.append(serialized(state))

        discover(
            "chess",
            self.positions,
            client,
            config=self.config,
            initial_library=self.initial,
            on_checkpoint=checkpoint,
            on_iteration=lambda event, current: calls.append("iteration"),
        )
        self.assertEqual(calls, ["checkpoint", "proposal", "checkpoint", "iteration"])
        self.assertEqual(states[0]["schema_version"], 1)
        self.assertEqual(states[0]["next_iteration"], 0)
        self.assertEqual(states[0]["proposals"], 0)
        self.assertEqual(states[0]["event_count"], 0)
        self.assertEqual(states[0]["history"], [])
        self.assertEqual(states[-1]["next_iteration"], 1)

    def test_nontext_provider_response_leaves_a_resumable_checkpoint(self):
        for response in (["not text"], {"not": "text"}, 1):
            client = OfflineClient([response])
            states = []
            with (
                self.subTest(response=response),
                self.assertRaisesRegex(ValueError, "proposal client must return text"),
            ):
                discover(
                    "chess",
                    self.positions,
                    client,
                    config=self.config,
                    initial_library=self.initial,
                    on_checkpoint=lambda state: states.append(serialized(state)),
                )
            failed = states[-1]
            self.assertEqual(failed["last_event"]["reason"], "provider_error")
            self.assertEqual(failed["next_iteration"], 0)
            self.assertEqual(failed["proposals"], 1)
            self.assertEqual(failed["history"], [])
            result, messages, _, resumed = self.run_segment(
                ["merged"], resume_state=failed
            )
            self.assertEqual(messages, client.messages)
            self.assertEqual([motif.name for motif in result.motifs], ["merged"])
            self.assertEqual(resumed[-1]["proposals"], 2)

    def test_changed_position_fields_or_order_fail_before_checkpoint_or_request(self):
        _, _, _, states = self.run_segment(["bad"])
        state = states[-1]
        altered = {
            "order": list(reversed(self.positions)),
            **{
                field: [
                    replace(self.positions[0], **{field: value}),
                    *self.positions[1:],
                ]
                for field, value in (
                    ("board", "changed-board"),
                    ("color", "black"),
                    ("label", 0),
                    ("id", "changed-id"),
                )
            },
        }
        for name, positions in altered.items():
            client, checkpoints = OfflineClient([]), []
            with self.subTest(field=name), self.assertRaises(ValueError):
                discover(
                    "chess",
                    positions,
                    client,
                    config=self.config,
                    resume_state=serialized(state),
                    on_checkpoint=checkpoints.append,
                )
            self.assertEqual(client.messages, [])
            self.assertEqual(checkpoints, [])

    def test_changed_semantic_config_fails_before_checkpoint_or_request(self):
        _, _, _, states = self.run_segment(["bad"])
        state = states[-1]
        changes = {
            "seed": 72,
            "min_delta": 0.02,
            "max_containment": 0.8,
            "max_motifs": 4,
            "timeout": 61,
            "max_p95_ms": 3,
            "examples_per_label": 1,
            "feedback_examples": 1,
            "history_turns": 1,
            "max_message_chars": 1000,
        }
        for name, value in changes.items():
            client, checkpoints = OfflineClient([]), []
            with self.subTest(field=name), self.assertRaises(ValueError):
                discover(
                    "chess",
                    self.positions,
                    client,
                    config=replace(self.config, **{name: value}),
                    resume_state=serialized(state),
                    on_checkpoint=checkpoints.append,
                )
            self.assertEqual(client.messages, [])
            self.assertEqual(checkpoints, [])

    def test_invalid_state_and_initial_plus_resume_are_rejected(self):
        _, _, _, states = self.run_segment(["bad"])
        state = states[-1]
        invalid = []
        for key, value in (
            ("schema_version", 999),
            ("next_iteration", -1),
            ("proposals", -1),
            ("history", "not a message list"),
            ("game", "go"),
        ):
            damaged = serialized(state)
            damaged[key] = value
            invalid.append((key, damaged, None))
        missing = serialized(state)
        del missing["library"]
        invalid.append(("missing library", missing, None))
        invalid.append(("initial plus resume", state, self.initial))
        for name, damaged, initial in invalid:
            client, checkpoints = OfflineClient([]), []
            with self.subTest(case=name), self.assertRaises(ValueError):
                discover(
                    "chess",
                    self.positions,
                    client,
                    config=self.config,
                    resume_state=damaged,
                    initial_library=initial,
                    on_checkpoint=checkpoints.append,
                )
            self.assertEqual(client.messages, [])
            self.assertEqual(checkpoints, [])

    def test_zero_proposal_checkpoint_preserves_original_initial_library(self):
        _, _, _, states = self.run_segment([])
        state = states[-1]
        self.assertEqual(state["next_iteration"], 0)
        self.assertEqual(state["initial_motifs"], 2)
        self.assertEqual(state["proposals"], 0)
        result, _, _, resumed = self.run_segment(["merged"], resume_state=state)
        provenance = result.metadata["initial_library"]
        self.assertEqual(provenance["metadata"], self.initial.metadata)
        self.assertEqual(
            provenance["sha256"],
            sha256(
                json.dumps(
                    library_to_dict(self.initial),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )
        self.assertEqual([motif.name for motif in result.motifs], ["merged"])
        self.assertEqual(resumed[-1]["accepted_proposals"], 1)


if __name__ == "__main__":
    unittest.main()
