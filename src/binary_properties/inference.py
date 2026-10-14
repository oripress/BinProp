from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .types import Evaluation, Library, Position

Evaluator = Callable[..., Evaluation]


def infer(
    library: Library,
    positive: Sequence[Position],
    negative: Sequence[Position],
    *,
    top_k: int = 3,
    min_fit: float = 0.5,
    timeout: float = 60.0,
    evaluate: Evaluator | None = None,
) -> dict[str, Any]:
    """Rank by fit times delta; fall back to one motif if none meets min_fit.

    Timeout applies per motif. Execution failures are returned in errors.
    """
    if library.game not in {"chess", "go"}:
        raise ValueError("game must be chess or go")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(min_fit) or not 0.0 <= min_fit <= 1.0:
        raise ValueError("min_fit must be finite and between zero and one")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    positive = tuple(positive)
    negative = tuple(negative)
    if not positive or not negative:
        raise ValueError("positive and negative support must both be nonempty")
    if evaluate is None:
        from .runtime import evaluate_source

        evaluate = evaluate_source

    # Keep evaluation independent of labels or any identifiers carrying metadata.
    support = [
        Position(board=position.board, color=position.color)
        for position in (*positive, *negative)
    ]
    ranked: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for motif in library.motifs:
        try:
            if not math.isfinite(motif.delta):
                raise ValueError("motif delta must be finite")
            if motif.delta <= 0:
                continue
            evaluation = evaluate(library.game, motif.source, support, timeout=timeout)
            if (
                evaluation.name != motif.name
                or evaluation.description != motif.description
            ):
                raise ValueError("motif metadata does not match its source")
            values = list(evaluation.values)
            if len(values) != len(support):
                raise ValueError("motif output count does not match the support count")
            if any(type(value) is not bool for value in values):
                raise ValueError("motif outputs must all be booleans")
            positive_values = values[: len(positive)]
            negative_values = values[len(positive) :]
            positive_true_rate = sum(positive_values) / len(positive_values)
            negative_false_rate = sum(not value for value in negative_values) / len(
                negative_values
            )
            support_fit = min(positive_true_rate, negative_false_rate)
            ranked.append(
                {
                    "id": motif.id,
                    "name": motif.name,
                    "description": motif.description,
                    "source": motif.source,
                    "delta": motif.delta,
                    "positive_true_rate": positive_true_rate,
                    "negative_false_rate": negative_false_rate,
                    "support_fit": support_fit,
                    "weighted_score": support_fit * motif.delta,
                }
            )
        except Exception as error:
            errors.append(
                {
                    "id": motif.id,
                    "name": motif.name,
                    "type": type(error).__name__,
                    "error": str(error),
                }
            )

    ranked.sort(
        key=lambda item: (
            -item["weighted_score"],
            -item["support_fit"],
            -item["delta"],
            item["name"],
        )
    )
    qualifying = [item for item in ranked if item["support_fit"] >= min_fit]
    selected = qualifying[:top_k] if qualifying else ranked[:1]
    return {
        "game": library.game,
        "policy": "support-fit-delta-v1",
        "top_k": top_k,
        "min_fit": min_fit,
        "support_counts": {"positive": len(positive), "negative": len(negative)},
        "selected": selected,
        "ranked": ranked,
        "qualifying_count": len(qualifying),
        "used_fallback": bool(selected and not qualifying),
        "errors": errors,
    }
