"""Held-out scoring with the paper's support-only selection policies."""

import ast
import re
from statistics import fmean

from binary_properties.inference import infer
from binary_properties.runtime import evaluate_source

from .runtime import evaluate_direct, helper_names

GROUPS = ("positive", "negative", "nearby_negative")


def rates(values, sizes):
    result, offset = {}, 0
    for group, size in zip(GROUPS, sizes):
        target = group == "positive"
        part = values[offset : offset + size]
        if len(part) != size or not size:
            raise ValueError("Every hidden group must be nonempty")
        result[group] = sum(v is target for v in part) / size
        offset += size
    if offset != len(values):
        raise ValueError("Prediction count does not match hidden positions")
    return result


def aggregate(episode, parts):
    return {
        "id": episode.id,
        "score": fmean(p["score"] for p in parts) if parts else 0.0,
        "motif_count": len(parts),
        "motifs": parts,
    }


def hidden_score(episode, candidates, evaluate):
    positions = [p for group in GROUPS for p in episode.hidden[group]]
    sizes = [len(episode.hidden[group]) for group in GROUPS]
    parts = []
    for candidate in candidates:
        try:
            values = evaluate(candidate, positions)
            component_rates = rates(values, sizes)
            invalid = any(type(value) is not bool for value in values)
            error = "invalid_outputs" if invalid else None
            score = min(component_rates.values())
        except Exception as exc:
            component_rates = {group: 0.0 for group in GROUPS}
            error, score, invalid = f"{type(exc).__name__}: {exc}", 0.0, True
        parts.append(
            {
                **candidate,
                "rates": component_rates,
                "score": score,
                "error": error,
                "invalid_outputs": invalid,
            }
        )
    return aggregate(episode, parts)


def score_binprop(episode, library, *, timeout=60):
    if library.game != episode.game:
        raise ValueError("Library and episode games must match")
    selected = infer(library, episode.positive, episode.negative, timeout=timeout)
    candidates = [
        {
            "name": row["name"],
            "source": row["source"],
            "support_fit": row["support_fit"],
        }
        for row in selected["selected"]
    ]
    result = hidden_score(
        episode,
        candidates,
        lambda c, positions: (
            evaluate_source(
                episode.game, c["source"], positions, timeout=timeout
            ).values
        ),
    )
    result.update(
        method="binprop",
        used_fallback=selected["used_fallback"],
        errors=selected["errors"],
    )
    return result


def extract_code(response):
    candidates = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", response, re.S | re.I)
    candidates.append(response)
    for source in reversed(candidates):
        try:
            tree = ast.parse(source.strip())
        except SyntaxError:
            continue
        if any(
            isinstance(n, ast.FunctionDef)
            and re.match(r"(?:explain|motif|basis)", n.name)
            for n in tree.body
        ):
            return source.strip()
    return None


def select(candidates):
    qualifying = [c for c in candidates if c["support_fit"] >= 0.5]
    return (
        qualifying or sorted(candidates, key=lambda c: -c["support_fit"])[:1],
        bool(candidates and not qualifying),
    )


def direct_candidates(episode, source, *, timeout=60):
    # Reject whole-board lookup without revealing hidden boards to the LM.
    if episode.game == "chess" and re.search(
        r"(?:[prnbqkPRNBQK1-8]+/){7}[prnbqkPRNBQK1-8]+", source
    ):
        return [], ["forbidden_fen_literal"]
    candidates, errors = [], []
    support = (*episode.positive, *episode.negative)
    try:
        names = helper_names(source, episode.game)
    except SyntaxError:
        return [], ["no_code"]
    if not names:
        return [], ["no_requested_helpers"]
    for name in names:
        try:
            values = evaluate_direct(
                episode.game, source, name, support, timeout=timeout
            )
            fit = min(
                sum(v is True for v in values[:4]) / 4,
                sum(v is False for v in values[4:]) / 4,
            )
            if episode.game == "chess" and any(v is None for v in values):
                fit = 0.0
            candidates.append({"name": name, "source": source, "support_fit": fit})
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    return candidates, errors


def score_direct(episode, source, *, timeout=60):
    candidates, errors = direct_candidates(episode, source, timeout=timeout)
    evaluated = hidden_score(
        episode,
        candidates,
        lambda c, positions: evaluate_direct(
            episode.game, c["source"], c["name"], positions, timeout=timeout
        ),
    )
    parts = evaluated["motifs"]
    if episode.game == "chess":
        # Preserve the original chess evaluator's invalid-helper exclusion.
        errors.extend(
            f"{p['name']}: {p['error']}" for p in parts if p["invalid_outputs"]
        )
        parts = [p for p in parts if not p["invalid_outputs"]]
    selected, fallback = select(parts)
    result = aggregate(episode, selected)
    result.update(method="direct", used_fallback=fallback, errors=errors)
    return result
