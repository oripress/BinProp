"""Regression cases for the paper's shared support-only retrieval policy."""

import unittest

from binary_properties.inference import infer
from binary_properties.types import Evaluation, Library, Motif, Position


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.positive = [
            Position("positive-1", "white", 1),
            Position("positive-2", "white", 1),
        ]
        self.negative = [
            Position("negative-1", "white", 0),
            Position("negative-2", "white", 0),
        ]

    def evaluate(self, values, calls=None):
        def run(game, source, positions, *, timeout):
            if calls is not None:
                calls.append((game, source, positions, timeout))
            result = values[source]
            if isinstance(result, Exception):
                raise result
            if isinstance(result, Evaluation):
                return result
            return Evaluation(
                result, source, "Original description: " + source, 0.0, 0.0
            )

        return run

    def motif(self, name, delta=1.0):
        return Motif(name, "Original description: " + name, name, delta)

    def test_weighted_rank_threshold_and_top_three_match_paper_policy(self):
        motifs = [
            self.motif("high-delta-low-fit", 4),
            self.motif("best", 0.9),
            self.motif("second", 0.8),
            self.motif("third", 0.7),
            self.motif("fourth", 0.6),
        ]
        values = {"high-delta-low-fit": [False, False, False, False]}
        values.update({m.source: [True, True, False, False] for m in motifs[1:]})
        result = infer(
            Library("chess", motifs),
            self.positive,
            self.negative,
            evaluate=self.evaluate(values),
        )
        self.assertEqual(
            [m["name"] for m in result["selected"]], ["best", "second", "third"]
        )
        self.assertEqual(result["qualifying_count"], 4)
        self.assertFalse(result["used_fallback"])
        self.assertEqual(result["selected"][0]["weighted_score"], 0.9)

    def test_threshold_is_inclusive_and_filters_before_top_k(self):
        motifs = [self.motif("below-threshold", 3), self.motif("at-threshold", 1)]
        pos = self.positive + [
            Position("positive-3", "white"),
            Position("positive-4", "white"),
        ]
        neg = self.negative + [
            Position("negative-3", "white"),
            Position("negative-4", "white"),
        ]
        values = {
            "below-threshold": [True, False, False, False, False, False, False, False],
            "at-threshold": [True, True, False, False, False, False, False, False],
        }
        result = infer(
            Library("chess", motifs), pos, neg, top_k=1, evaluate=self.evaluate(values)
        )
        self.assertEqual(result["ranked"][0]["name"], "below-threshold")
        self.assertEqual(result["selected"][0]["name"], "at-threshold")
        self.assertEqual(result["selected"][0]["support_fit"], 0.5)

    def test_no_qualifying_motif_returns_exactly_one_not_top_three(self):
        motifs = [self.motif("alpha", 0.1), self.motif("beta", 0.9)]
        values = {m.source: [False, False, True, True] for m in motifs}
        result = infer(
            Library("chess", motifs),
            self.positive,
            self.negative,
            evaluate=self.evaluate(values),
        )
        self.assertEqual([m["name"] for m in result["selected"]], ["beta"])
        self.assertEqual(result["qualifying_count"], 0)
        self.assertTrue(result["used_fallback"])

    def test_tie_break_uses_fit_then_delta_then_name_and_stable_input_order(self):
        motifs = [
            self.motif("zeta", 0.5),
            self.motif("alpha", 0.5),
            self.motif("lower-fit", 1),
        ]
        values = {
            "zeta": [True, True, False, False],
            "alpha": [True, True, False, False],
            "lower-fit": [True, False, False, False],
        }
        result = infer(
            Library("chess", motifs),
            self.positive,
            self.negative,
            evaluate=self.evaluate(values),
        )
        self.assertEqual(
            [m["name"] for m in result["selected"]], ["alpha", "zeta", "lower-fit"]
        )
        first = Motif("same", "first unchanged", "first", 0.5)
        second = Motif("same", "second unchanged", "second", 0.5)
        result = infer(
            Library("chess", [first, second]),
            self.positive,
            self.negative,
            evaluate=self.evaluate(
                {
                    "first": Evaluation(
                        [True, True, False, False], "same", "first unchanged", 0, 0
                    ),
                    "second": Evaluation(
                        [True, True, False, False], "same", "second unchanged", 0, 0
                    ),
                }
            ),
        )
        self.assertEqual(
            [m["description"] for m in result["selected"]],
            ["first unchanged", "second unchanged"],
        )

    def test_nonpositive_delta_is_excluded_without_reorienting_predicate(self):
        motifs = [self.motif("inverse", -0.9), self.motif("zero", 0)]
        calls = []
        result = infer(
            Library("chess", motifs),
            self.positive,
            self.negative,
            evaluate=self.evaluate({}, calls),
        )
        self.assertEqual(result["selected"], [])
        self.assertEqual(calls, [])
        self.assertFalse(result["used_fallback"])

    def test_runtime_failure_and_invalid_values_are_explicit_not_false_predictions(
        self,
    ):
        motifs = [
            self.motif(name)
            for name in ("timeout", "invalid", "short", "nonbool", "good")
        ]
        values = {
            "timeout": TimeoutError("timed out"),
            "invalid": ValueError("invalid source"),
            "short": [True],
            "nonbool": [1, True, False, False],
            "good": [True, True, False, False],
        }
        result = infer(
            Library("chess", motifs),
            self.positive,
            self.negative,
            evaluate=self.evaluate(values),
        )
        self.assertEqual([m["name"] for m in result["selected"]], ["good"])
        self.assertEqual(
            [e["name"] for e in result["errors"]],
            ["timeout", "invalid", "short", "nonbool"],
        )
        self.assertEqual(result["errors"][0]["type"], "TimeoutError")
        self.assertEqual(result["selected"][0]["negative_false_rate"], 1)

    def test_original_metadata_is_preserved_and_labels_cannot_change_selection(self):
        motif = Motif(
            "original name", "Original model description.", "original source", 0.8
        )
        values = {
            motif.source: Evaluation(
                [True, False, False, False],
                motif.name,
                motif.description,
                0,
                0,
            )
        }
        calls = []
        library = Library("chess", [motif], {"hidden_score": 1.0})
        result = infer(
            library,
            self.positive,
            self.negative,
            timeout=12,
            evaluate=self.evaluate(values, calls),
        )
        flipped_pos = [
            Position(p.board, p.color, 0, "hidden-id") for p in self.positive
        ]
        flipped_neg = [
            Position(p.board, p.color, 1, "hidden-id") for p in self.negative
        ]
        changed = infer(
            Library("chess", [motif], {"hidden_score": -9}),
            flipped_pos,
            flipped_neg,
            timeout=12,
            evaluate=self.evaluate(values, calls),
        )
        self.assertEqual(result, changed)
        selected = result["selected"][0]
        self.assertEqual(
            (
                selected["id"],
                selected["name"],
                selected["description"],
                selected["source"],
            ),
            (motif.id, motif.name, motif.description, motif.source),
        )
        self.assertEqual(selected["positive_true_rate"], 0.5)
        self.assertEqual(selected["negative_false_rate"], 1)
        self.assertEqual(calls[0][3], 12)
        self.assertTrue(
            all(p.label is None and p.id == "" for call in calls for p in call[2])
        )

    def test_go_and_chess_use_identical_selection_policy(self):
        motifs = [self.motif("good", 0.6)]
        values = {"good": [True, True, False, False]}
        chess = infer(
            Library("chess", motifs),
            self.positive,
            self.negative,
            evaluate=self.evaluate(values),
        )
        go_pos = [
            Position(tuple(["B" + "." * 18] + ["." * 19] * 18), "B") for _ in range(2)
        ]
        go_neg = [Position(tuple(["." * 19] * 19), "B") for _ in range(2)]
        go = infer(
            Library("go", motifs), go_pos, go_neg, evaluate=self.evaluate(values)
        )
        self.assertEqual(go["selected"], chess["selected"])
        self.assertEqual(go["game"], "go")

    def test_empty_library_and_invalid_options(self):
        library = Library("chess")
        result = infer(
            library, self.positive, self.negative, evaluate=self.evaluate({})
        )
        self.assertEqual(result["selected"], [])
        for positive, negative, kwargs in [
            ([], self.negative, {}),
            (self.positive, [], {}),
            (self.positive, self.negative, {"top_k": 0}),
            (self.positive, self.negative, {"top_k": True}),
            (self.positive, self.negative, {"min_fit": float("nan")}),
            (self.positive, self.negative, {"min_fit": 1.1}),
            (self.positive, self.negative, {"timeout": float("inf")}),
            (self.positive, self.negative, {"timeout": 0}),
        ]:
            with self.subTest(
                kwargs=kwargs, positive=len(positive), negative=len(negative)
            ):
                with self.assertRaises(ValueError):
                    infer(
                        library,
                        positive,
                        negative,
                        evaluate=self.evaluate({}),
                        **kwargs,
                    )
        result = infer(
            Library("chess", [self.motif("invalid", float("nan"))]),
            self.positive,
            self.negative,
            evaluate=self.evaluate({}),
        )
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["errors"][0]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
