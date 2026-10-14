from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import chess

from .games.common import validate_motif_text
from .types import Game, Library, Motif, Position


def position_from_dict(
    value: dict, game: Game, *, require_label: bool = False, default_id: str = ""
) -> Position:
    if not isinstance(value, dict):
        raise ValueError("Each position must be a JSON object")
    label = value.get("label")
    if label is not None:
        if type(label) is str and label in ("winning", "losing"):
            label = int(label == "winning")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be 0, 1, 'winning', or 'losing'")
    if require_label and label is None:
        raise ValueError("Discovery positions must have a label")
    identifier = value.get("id", default_id)
    if not isinstance(identifier, str):
        raise ValueError("id must be a string")
    if game == "chess":
        fen = value.get("fen")
        if not isinstance(fen, str) or len(fen.split()) != 6:
            raise ValueError("Chess positions require a complete six-field FEN")
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        color = value.get("color", "white" if board.turn else "black")
        if not isinstance(color, str) or color.lower() not in (
            "w",
            "white",
            "b",
            "black",
        ):
            raise ValueError("Chess color must be white or black")
        color = "white" if color.lower() in ("w", "white") else "black"
        return Position(fen, color, label, identifier)
    if game == "go":
        rows = value.get("board_rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 19
            or any(
                not isinstance(row, str) or len(row) != 19 or set(row) - set(".BW")
                for row in rows
            )
        ):
            raise ValueError("Go requires 19 board_rows of 19 characters from .BW")
        color = value.get("color")
        if not isinstance(color, str) or color.upper() not in (
            "B",
            "BLACK",
            "W",
            "WHITE",
        ):
            raise ValueError("Go color must be explicitly B or W")
        color = "B" if color.upper() in ("B", "BLACK") else "W"
        return Position(tuple(rows), color, label, identifier)
    raise ValueError("game must be chess or go")


def position_to_dict(position: Position, game: Game) -> dict:
    result = (
        {"fen": position.board}
        if game == "chess"
        else {"board_rows": list(position.board)}
    )
    result["color"] = position.color
    if position.label is not None:
        result["label"] = position.label
    if position.id:
        result["id"] = position.id
    return result


def load_positions(
    path: str | Path, game: Game, *, require_labels: bool = True
) -> list[Position]:
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append(
                    position_from_dict(
                        json.loads(line),
                        game,
                        require_label=require_labels,
                        default_id=str(number),
                    )
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid position on line {number}: {exc}") from exc
    if not result:
        raise ValueError("The position file is empty")
    if require_labels and {position.label for position in result} != {0, 1}:
        raise ValueError("Discovery requires both winning and losing positions")
    return result


def load_groups(path: str | Path, game: Game) -> tuple[list[Position], list[Position]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Comparison must be a JSON object")
    groups = []
    for key in ("positive", "negative"):
        rows = value.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} must be a nonempty list of positions")
        groups.append(
            [
                position_from_dict(row, game, default_id=f"{key}-{i}")
                for i, row in enumerate(rows)
            ]
        )
    return groups[0], groups[1]


def library_to_dict(library: Library) -> dict:
    return {
        "schema_version": 1,
        "game": library.game,
        "motifs": [{"id": motif.id, **asdict(motif)} for motif in library.motifs],
        "metadata": library.metadata,
    }


def load_library(path: str | Path) -> Library:
    return library_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def library_from_dict(value: object) -> Library:
    """Validate a serialized library without changing its original sources."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported motif library schema")
    game = value.get("game")
    if game not in ("chess", "go"):
        raise ValueError("Library game must be chess or go")
    if not isinstance(value.get("motifs"), list) or not isinstance(
        value.get("metadata", {}), dict
    ):
        raise ValueError("Invalid motif library")
    motifs, identifiers = [], set()
    for row in value["motifs"]:
        if not isinstance(row, dict):
            raise ValueError("Each motif must be a JSON object")
        for key in ("name", "description", "source"):
            validate_motif_text(row.get(key), key)
        delta = row.get("delta")
        if (
            type(delta) not in (int, float)
            or not math.isfinite(delta)
            or not -1 <= delta <= 1
        ):
            raise ValueError("Motif delta must be finite and between -1 and 1")
        motif = Motif(row["name"], row["description"], row["source"], float(delta))
        if row.get("id", motif.id) != motif.id:
            raise ValueError("Motif source hash does not match its id")
        if motif.id in identifiers:
            raise ValueError("Duplicate motif source in library")
        identifiers.add(motif.id)
        motifs.append(motif)
    return Library(game, motifs, value.get("metadata", {}))


def write_json(path: str | Path, value: object) -> None:
    """Write JSON atomically; the parent directory must exist."""
    destination = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix="." + destination.name,
            delete=False,
        ) as handle:
            name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, destination)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def save_library(path: str | Path, library: Library) -> None:
    value = library_to_dict(library)
    library_from_dict(value)
    write_json(path, value)
