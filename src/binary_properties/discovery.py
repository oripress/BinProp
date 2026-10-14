"""Discover motifs on labeled positions using activation gaps and overlap."""

from __future__ import annotations

import json
import math
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Callable, Protocol, Sequence

from .games.common import MOTIF_FIELD_LIMITS, validate_motif_text
from .io import library_from_dict, library_to_dict
from .runtime import evaluate_source
from .types import Evaluation, Game, Library, Motif, Position


class ProposalClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class DiscoveryConfig:
    iterations: int = 100
    min_delta: float = 0.01
    max_containment: float = 0.70
    max_motifs: int = 0
    timeout: float = 60
    max_p95_ms: float = 2
    seed: int = 0
    examples_per_label: int = 2
    feedback_examples: int = 2
    history_turns: int = 6
    max_message_chars: int = 16000

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "max_motifs",
            "examples_per_label",
            "feedback_examples",
            "history_turns",
            "max_message_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.max_message_chars < 100:
            raise ValueError("max_message_chars must be at least 100")
        for name in ("min_delta", "max_containment", "timeout", "max_p95_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.min_delta < 1:
            raise ValueError("min_delta must lie in [0, 1)")
        if not 0 < self.max_containment <= 1:
            raise ValueError("max_containment must lie in (0, 1]")
        if self.timeout <= 0 or self.max_p95_ms <= 0:
            raise ValueError("timeout and max_p95_ms must be positive")


def _labels(positions: Sequence[Position]) -> list[int]:
    labels = [position.label for position in positions]
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError(
            "discovery requires an explicit integer label 0 or 1 on every position"
        )
    if set(labels) != {0, 1}:
        raise ValueError("discovery requires both label classes")
    return labels


def _check_evaluation(evaluation: Evaluation, count: int) -> None:
    if len(evaluation.values) != count or any(
        type(value) is not bool for value in evaluation.values
    ):
        raise ValueError("runtime must return one Boolean value per position")
    validate_motif_text(evaluation.name, "name")
    validate_motif_text(evaluation.description, "description")
    for timing in (evaluation.p95_ms, evaluation.max_ms):
        if not math.isfinite(timing) or timing < 0:
            raise ValueError("runtime returned invalid timing")


def activation_stats(
    values: Sequence[bool], labels: Sequence[int]
) -> dict[str, float | int]:
    """Class-conditional activation rates and their difference."""
    if len(values) != len(labels) or not values:
        raise ValueError("values and labels must have equal nonzero length")
    if any(type(value) is not bool for value in values):
        raise ValueError("values must be Boolean")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("labels must be integer 0 or 1")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("both label classes are required")
    positive_true = sum(value for value, label in zip(values, labels) if label == 1)
    negative_true = sum(value for value, label in zip(values, labels) if label == 0)
    positive_rate = positive_true / positive_count
    negative_rate = negative_true / negative_count
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_true": positive_true,
        "negative_true": negative_true,
        "p_true_positive": positive_rate,
        "p_true_negative": negative_rate,
        "delta": positive_rate - negative_rate,
        "activation_rate": sum(values) / len(values),
    }


def winning_containment(
    left: Sequence[bool],
    right: Sequence[bool],
    labels: Sequence[int],
) -> float:
    """Maximum directional containment, considering only label-1 rows."""
    if len(left) != len(right) or len(left) != len(labels):
        raise ValueError("containment inputs must have equal length")
    a = b = intersection = 0
    for lvalue, rvalue, label in zip(left, right, labels):
        if label == 1:
            a += bool(lvalue)
            b += bool(rvalue)
            intersection += bool(lvalue and rvalue)
    return max(intersection / a if a else 0, intersection / b if b else 0)


def _extract_source(response: str) -> str:
    # Preserve bytes inside the fence, including literal names and newlines.
    blocks = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(blocks) != 1:
        raise ValueError("return exactly one fenced Python code block")
    return validate_motif_text(blocks[0], "source")


def _example(position: Position, index: int, **extra: object) -> dict:
    return {
        "index": index,
        "id": position.id,
        "board": position.board,
        "color": position.color,
        "label": position.label,
        **extra,
    }


def _feedback_examples(
    positions: Sequence[Position],
    values: Sequence[bool],
    limit: int,
    overlap_values: Sequence[bool] | None = None,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {
        "positive_misses": [],
        "negative_activations": [],
    }
    if overlap_values is not None:
        groups["winning_overlap"] = []
        groups["winning_disagreement"] = []
    for i, (position, value) in enumerate(zip(positions, values)):
        kinds = []
        if position.label == 1 and not value:
            kinds.append("positive_misses")
        if position.label == 0 and value:
            kinds.append("negative_activations")
        if overlap_values is not None and position.label == 1:
            if value and overlap_values[i]:
                kinds.append("winning_overlap")
            elif value != overlap_values[i]:
                kinds.append("winning_disagreement")
        for kind in kinds:
            if len(groups[kind]) < limit:
                extra = {"candidate_value": value}
                if overlap_values is not None:
                    extra["incumbent_value"] = overlap_values[i]
                groups[kind].append(_example(position, i, **extra))
    return groups


def _system_prompt(game: Game, config: DiscoveryConfig) -> str:
    if game == "chess":
        contract = (
            "Define motif(fen: str) -> bool. Parse with chess.Board(fen). "
            "The FEN side to move is the evaluated side; use board.turn for it "
            "and not board.turn for its opponent. The predicate must be invariant "
            "under chess.Board.mirror() with colors swapped. You may import chess "
            "and math. board.piece_at(square) is a Piece or None; compare its "
            ".piece_type, not the Piece itself, to chess.PAWN etc."
        )
    else:
        contract = (
            "Define motif(board_rows: list[str], color: str) -> bool. "
            "board_rows contains 19 top-to-bottom strings of 19 characters: "
            "'.' empty, 'B' black, 'W' white. color is the evaluated side, 'B' or "
            "'W'. Use this explicit color argument. White receives komi, so "
            "color-swapped results need not be identical. "
            "You may import math and sgfmill."
        )
    delta_rule = ">=" if game == "chess" else ">"
    containment_rule = "<" if game == "chess" and config.max_containment < 1 else "<="
    return (
        f"Discover concise, meaningful Boolean properties of {game} positions. "
        "Label 1 is the positive (winning) class, label 0 the comparison class. "
        "Do not assume score thresholds: labels are supplied by the user. "
        "True should describe an advantage for the evaluated player.\n"
        + contract
        + "\nReturn exactly one fenced Python code block with literal strings "
        "MOTIF_NAME and MOTIF_DESCRIPTION and the function. No prose or tests. "
        f"Use at most {MOTIF_FIELD_LIMITS['name']} characters for MOTIF_NAME, "
        f"{MOTIF_FIELD_LIMITS['description']} for MOTIF_DESCRIPTION, and "
        f"{MOTIF_FIELD_LIMITS['source']} source characters. "
        "No engines, files, network, "
        "randomness, mutable global state, recursion, classes, or dynamic calls. "
        "Terminate quickly.\n"
        f"Acceptance requires delta=P(True|1)-P(True|0) {delta_rule} "
        f"{config.min_delta}. Winning-only overlap is "
        "max(intersection/candidate_true, intersection/incumbent_true), "
        f"and must be {containment_rule} {config.max_containment}. "
        "A proposal above the overlap limit can replace all overlapping motifs "
        "only when its delta is strictly greater than every removed motif. "
        f"Runtime p95 must be at most {config.max_p95_ms} ms. "
        + (
            "Overall activation fraction must lie in [0.005,0.95]. "
            if game == "chess"
            else ""
        )
        + "Use concrete evaluator feedback to revise rejected concepts."
    )


def _current_prompt(
    game: Game,
    positions: Sequence[Position],
    motifs: Sequence[Motif],
    outputs: Sequence[Sequence[bool]],
    iteration: int,
    config: DiscoveryConfig,
) -> str:
    rng = random.Random(config.seed + iteration * 1009)
    examples = []
    for label in (1, 0):
        indices = [i for i, position in enumerate(positions) if position.label == label]
        rng.shuffle(indices)
        for i in indices[: config.examples_per_label]:
            examples.append(
                _example(
                    positions[i],
                    i,
                    active_motifs=[
                        motif.name
                        for motif, values in zip(motifs, outputs)
                        if values[i]
                    ],
                )
            )
    summary = [
        {
            "id": motif.id,
            "name": motif.name,
            "description": motif.description,
            "delta": motif.delta,
        }
        for motif in sorted(motifs, key=lambda motif: (-motif.delta, motif.name))
    ]
    payload = {
        "iteration": iteration,
        "game": game,
        "current_motifs": summary,
        "labeled_examples": examples,
        "task": "Propose one useful new motif or an improved replacement.",
    }
    if config.max_motifs and len(motifs) >= config.max_motifs:
        payload["weakest_motif"] = min(motifs, key=lambda motif: motif.delta).name
    return json.dumps(payload, ensure_ascii=False)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _resume_library(state: dict, game: Game, config: dict, fingerprint: str) -> Library:
    """Validate saved progress before evaluating code or contacting the model."""
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported discovery checkpoint schema")
    if state.get("game") != game or state.get("positions_sha256") != fingerprint:
        raise ValueError("Resume game or positions do not match the checkpoint")
    saved_config = state.get("config", {})
    if not isinstance(saved_config, dict) or {
        key: value for key, value in saved_config.items() if key != "iterations"
    } != {key: value for key, value in config.items() if key != "iterations"}:
        raise ValueError("Resume configuration does not match the checkpoint")
    for key in (
        "next_iteration",
        "initial_motifs",
        "proposals",
        "accepted_proposals",
        "event_count",
    ):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f"Invalid checkpoint {key}")
    if not (
        state["accepted_proposals"] <= state["next_iteration"] <= state["proposals"]
    ):
        raise ValueError("Inconsistent checkpoint counters")
    if state["event_count"] != state["proposals"]:
        raise ValueError("Inconsistent checkpoint event count")
    history = state.get("history")
    if (
        not isinstance(history, list)
        or len(history) % 2
        or len(history) > 2 * config["history_turns"]
    ):
        raise ValueError("Invalid checkpoint feedback history")
    for i, message in enumerate(history):
        if (
            not isinstance(message, dict)
            or message.get("role") != ("assistant" if i % 2 == 0 else "user")
            or not isinstance(message.get("content"), str)
            or len(message["content"]) > config["max_message_chars"] + 64
        ):
            raise ValueError("Invalid checkpoint feedback message")
    library = library_from_dict(state.get("library"))
    if library.game != game:
        raise ValueError("Checkpoint library game does not match")
    return library


def _evaluate_response(
    game: Game,
    response: str,
    positions: Sequence[Position],
    timeout: float,
    event: dict,
) -> Evaluation | None:
    try:
        source = _extract_source(response)
    except ValueError as exc:
        event.update(reason="invalid_response", error=str(exc))
        return None
    event["source"] = source
    try:
        evaluation = evaluate_source(game, source, positions, timeout=timeout)
        _check_evaluation(evaluation, len(positions))
    except Exception as exc:
        event.update(
            reason="runtime_error", error=f"{type(exc).__name__}: {exc}"[:1000]
        )
        return None
    return evaluation


def _select_replacements(
    game: Game,
    candidate: Motif,
    values: Sequence[bool],
    motifs: Sequence[Motif],
    outputs: Sequence[Sequence[bool]],
    labels: Sequence[int],
    *,
    config: DiscoveryConfig,
    event: dict,
) -> tuple[str | None, set[int], Sequence[bool] | None]:
    """Choose replacements without modifying the current library."""
    remove = set()
    overlaps = []
    for i, existing in enumerate(outputs):
        containment = winning_containment(values, existing, labels)
        over_cap = (
            containment >= config.max_containment
            if game == "chess" and config.max_containment < 1
            else containment > config.max_containment
        )
        if over_cap:
            remove.add(i)
            overlaps.append(
                {
                    "index": i,
                    "id": motifs[i].id,
                    "name": motifs[i].name,
                    "delta": motifs[i].delta,
                    "containment": containment,
                }
            )
    if overlaps:
        event["overlaps"] = overlaps

    # Exact/complement signatures stay forbidden when the overlap cap is disabled.
    if game == "chess":
        for i, existing in enumerate(outputs):
            if values == existing or all(a != b for a, b in zip(values, existing)):
                event["duplicate_of"] = motifs[i].name
                event["duplicate_of_id"] = motifs[i].id
                return "duplicate_signature", remove, existing

    blockers = [i for i in remove if candidate.delta <= motifs[i].delta]
    if blockers:
        blocker = max(blockers, key=lambda i: motifs[i].delta)
        event["blocking_motif"] = motifs[blocker].name
        event["blocking_motif_id"] = motifs[blocker].id
        return "overlap_not_improved", remove, outputs[blocker]

    if config.max_motifs and len(motifs) - len(remove) >= config.max_motifs:
        weakest = min(range(len(motifs)), key=lambda i: (motifs[i].delta, i))
        if candidate.delta <= motifs[weakest].delta:
            return "weakest_not_improved", remove, None
        remove.add(weakest)
    return None, remove, None


def discover(
    game: Game,
    positions: Sequence[Position],
    client: ProposalClient,
    *,
    config: DiscoveryConfig | None = None,
    initial_library: Library | None = None,
    on_iteration: Callable[[dict, Library], None] | None = None,
    resume_state: dict | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> Library:
    """Run config.iterations additional proposals, rescoring initial motifs.

    Checkpoints retain feedback for resume; libraries and events omit boards.
    """
    if game not in ("chess", "go"):
        raise ValueError(f"unsupported game: {game!r}")
    config = config or DiscoveryConfig()
    positions = list(positions)
    labels = _labels(positions)
    positive_count = sum(labels)
    fingerprint = _fingerprint([asdict(position) for position in positions])
    effective_config = asdict(config)
    if resume_state is not None:
        if initial_library is not None:
            raise ValueError("Use either initial_library or resume_state, not both")
        initial_library = _resume_library(
            resume_state, game, effective_config, fingerprint
        )
    if initial_library is not None and initial_library.game != game:
        raise ValueError("initial library game does not match discovery game")
    motifs: list[Motif] = []
    outputs: list[list[bool]] = []
    if initial_library is not None:
        if config.max_motifs and len(initial_library.motifs) > config.max_motifs:
            raise ValueError("initial library exceeds max_motifs")
        if len({motif.id for motif in initial_library.motifs}) != len(
            initial_library.motifs
        ):
            raise ValueError("Duplicate motif source in initial library")
        for motif in initial_library.motifs:
            evaluation = evaluate_source(
                game, motif.source, positions, timeout=config.timeout
            )
            _check_evaluation(evaluation, len(positions))
            if (
                evaluation.name != motif.name
                or evaluation.description != motif.description
            ):
                raise ValueError("initial library metadata does not match its source")
            delta = float(activation_stats(evaluation.values, labels)["delta"])
            if resume_state is not None and delta != motif.delta:
                raise ValueError("Resumed motif values differ from the checkpoint")
            motifs.append(Motif(motif.name, motif.description, motif.source, delta))
            outputs.append(evaluation.values)
    initial_count = len(motifs)
    accepted_count = attempted_count = next_iteration = event_count = 0
    history: list[dict[str, str]] = []
    last_event = None
    origin = {}
    if initial_library is not None and (
        initial_library.motifs or initial_library.metadata
    ):
        origin = {
            "sha256": _fingerprint(library_to_dict(initial_library)),
            "metadata": deepcopy(initial_library.metadata),
        }
    if resume_state is not None:
        initial_count = resume_state["initial_motifs"]
        accepted_count = resume_state["accepted_proposals"]
        attempted_count = resume_state["proposals"]
        next_iteration = resume_state["next_iteration"]
        event_count = resume_state["event_count"]
        history = deepcopy(resume_state["history"])
        last_event = deepcopy(resume_state.get("last_event"))
        origin = deepcopy(initial_library.metadata.get("initial_library", {}))
    target = next_iteration + config.iterations
    effective_config["iterations"] = target

    def snapshot() -> Library:
        metadata = {
            "discovery": {
                "config": effective_config,
                "positions": len(positions),
                "positive_count": positive_count,
                "negative_count": len(labels) - positive_count,
                "initial_motifs": initial_count,
                "proposals": attempted_count,
                "accepted_proposals": accepted_count,
                "method": "full_pool_activation_delta_winning_containment",
            },
        }
        if origin:
            metadata["initial_library"] = deepcopy(origin)
        return Library(game=game, motifs=list(motifs), metadata=metadata)

    def checkpoint() -> None:
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "schema_version": 1,
                    "game": game,
                    "config": dict(effective_config),
                    "positions_sha256": fingerprint,
                    "library": library_to_dict(snapshot()),
                    "next_iteration": next_iteration,
                    "initial_motifs": initial_count,
                    "proposals": attempted_count,
                    "accepted_proposals": accepted_count,
                    "event_count": event_count,
                    "last_event": deepcopy(last_event),
                    "history": deepcopy(history),
                }
            )

    checkpoint()
    system = {"role": "system", "content": _system_prompt(game, config)}
    for iteration in range(next_iteration, target):
        attempted_count += 1
        messages = [
            system,
            *history,
            {
                "role": "user",
                "content": _current_prompt(
                    game, positions, motifs, outputs, iteration, config
                ),
            },
        ]
        event: dict = {"iteration": iteration, "accepted": False}
        response = ""
        feedback: dict = {}
        provider_error: Exception | None = None
        evaluation = None
        try:
            response = client.complete(messages)
            if not isinstance(response, str):
                raise ValueError("proposal client must return text")
        except Exception as exc:
            provider_error = exc
            event.update(
                reason="provider_error",
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
        else:
            evaluation = _evaluate_response(
                game, response, positions, config.timeout, event
            )
        if evaluation is not None:
            source = event["source"]
            stats = activation_stats(evaluation.values, labels)
            delta = float(stats["delta"])
            candidate = Motif(evaluation.name, evaluation.description, source, delta)
            event.update(
                motif_id=candidate.id,
                name=evaluation.name,
                description=evaluation.description,
                delta=delta,
                stats=stats,
                p95_ms=evaluation.p95_ms,
                max_ms=evaluation.max_ms,
            )
            reason = None
            remove: set[int] = set()
            if evaluation.p95_ms > config.max_p95_ms:
                reason = "too_slow"
            elif game == "chess" and any(
                motif.name == evaluation.name for motif in motifs
            ):
                reason = "duplicate_name"
            elif any(motif.id == candidate.id for motif in motifs):
                reason = "duplicate_source"
            elif game == "chess" and not 0.005 <= stats["activation_rate"] <= 0.95:
                reason = "activation_rate_out_of_range"
            elif delta < config.min_delta or (
                game == "go" and delta == config.min_delta
            ):
                reason = "delta_too_low"
            comparison = None
            if reason is None:
                reason, remove, comparison = _select_replacements(
                    game,
                    candidate,
                    evaluation.values,
                    motifs,
                    outputs,
                    labels,
                    config=config,
                    event=event,
                )
            if reason is None:
                event["removed"] = [motifs[i].name for i in sorted(remove)]
                event["removed_ids"] = [motifs[i].id for i in sorted(remove)]
                motifs = [motif for i, motif in enumerate(motifs) if i not in remove]
                outputs = [
                    values for i, values in enumerate(outputs) if i not in remove
                ]
                motifs.append(candidate)
                outputs.append(evaluation.values)
                accepted_count += 1
                event.update(
                    accepted=True,
                    reason="accepted_replace" if remove else "accepted_add",
                )
            else:
                event["reason"] = reason
                feedback = _feedback_examples(
                    positions, evaluation.values, config.feedback_examples, comparison
                )
        event["basis_count"] = len(motifs)
        # Keep feedback boards out of decision logs and exported metadata.
        feedback_payload = {
            key: value for key, value in event.items() if key != "source"
        }
        if feedback:
            feedback_payload["examples"] = feedback
        if provider_error is None and response:
            history.extend(
                (
                    {
                        "role": "assistant",
                        "content": response[: config.max_message_chars],
                    },
                    {
                        "role": "user",
                        "content": "Evaluator feedback:\n"
                        + json.dumps(feedback_payload, ensure_ascii=False)[
                            : config.max_message_chars
                        ],
                    },
                )
            )
            history = (
                history[-2 * config.history_turns :] if config.history_turns else []
            )
        # A failed provider call retries this same prompt on resume.
        if provider_error is None:
            next_iteration = iteration + 1
        event_count += 1
        last_event = dict(event)
        checkpoint()
        if on_iteration is not None:
            on_iteration(dict(event), snapshot())
        if provider_error is not None:
            raise provider_error
    return snapshot()
