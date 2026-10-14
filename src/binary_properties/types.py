from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

Game = Literal["chess", "go"]


@dataclass(frozen=True)
class Position:
    board: str | tuple[str, ...]
    color: str
    label: int | None = None
    id: str = ""


@dataclass(frozen=True)
class Motif:
    name: str
    description: str
    source: str
    delta: float

    @property
    def id(self) -> str:
        return sha256(self.source.encode("utf-8")).hexdigest()[:16]


@dataclass
class Library:
    game: Game
    motifs: list[Motif] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    values: list[bool]
    name: str
    description: str
    p95_ms: float
    max_ms: float
