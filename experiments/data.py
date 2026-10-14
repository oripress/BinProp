"""Read explicit episodes or the paper's stored continuation-tree rows."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from binary_properties.io import position_from_dict
from binary_properties.types import Position


@dataclass(frozen=True)
class Episode:
    id: str
    game: str
    positive: tuple[Position, ...]
    negative: tuple[Position, ...]
    hidden: dict[str, tuple[Position, ...]]


def evenly_spaced(pool, count=4):
    if len(pool) < count:
        raise ValueError("Each support pool must have at least four positions")
    return tuple(pool[round(i * (len(pool) - 1) / (count - 1))] for i in range(count))


def episode_from_dict(row, game):
    if game not in {"chess", "go"} or not isinstance(row, dict):
        raise ValueError("Expected a chess or Go episode object")
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Each episode requires a nonempty string id")
    color = row.get("root_turn") if game == "chess" else row.get("root_to_play")

    def positions(items):
        if not isinstance(items, list):
            raise ValueError("Position groups must be JSON lists")
        result = []
        for item in items:
            value = dict(item)
            if color is not None:
                value["color"] = color
            result.append(position_from_dict(value, game))
        return tuple(result)

    if "support" in row:
        support, hidden = row["support"], row["held_out"]
        positive = positions(support["positive"])
        negative = positions(support["negative"])
        groups = {
            key: positions(hidden[key])
            for key in ("positive", "negative", "nearby_negative")
        }
    else:
        positive = evenly_spaced(positions(row["positive_support_pool"]))
        negative = evenly_spaced(positions(row["negative_support_pool"]))
        excluded = {p.board for p in (*positive, *negative)}
        groups = {}
        for key, stored in (
            ("positive", "positive_holdout"),
            ("negative", "normal_negative_holdout"),
            ("nearby_negative", "counterfactual_negative_holdout"),
        ):
            groups[key] = tuple(
                p for p in positions(row[stored]) if p.board not in excluded
            )
    if (
        len(positive) != 4
        or len(negative) != 4
        or any(len(group) != 8 for group in groups.values())
    ):
        raise ValueError(
            "Paper episodes require 4+4 support positions and 8 hidden positions in each of three groups"
        )
    all_positions = [
        *positive,
        *negative,
        *(p for group in groups.values() for p in group),
    ]
    if len({p.color for p in all_positions}) != 1:
        raise ValueError("Keep the evaluated player fixed throughout an episode")
    for p in all_positions:
        if game == "chess" and p.board.split()[1] != (
            "w" if p.color == "white" else "b"
        ):
            raise ValueError(
                "All chess continuations must have the evaluated player to move"
            )
    keys = [p.board for p in all_positions]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Support and hidden groups must be disjoint and contain distinct boards"
        )
    return Episode(identifier, game, positive, negative, groups)


def load_episodes(path, game):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    episodes = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                episodes.append(episode_from_dict(json.loads(line), game))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Invalid episode on line {line_number}: {error}"
                ) from error
    if not episodes or len({e.id for e in episodes}) != len(episodes):
        raise ValueError("Episodes must be nonempty and have unique ids")
    return episodes
