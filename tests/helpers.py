"""Shared offline clients and board fixtures."""

from copy import deepcopy

from binary_properties.types import Position


class OfflineClient:
    def __init__(self, responses, calls=None):
        self.responses = iter(responses)
        self.messages = []
        self.calls = calls

    def complete(self, messages):
        if self.calls is not None:
            self.calls.append("proposal")
        self.messages.append(deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def motif_source(name, *, description="Literal description"):
    return (
        f"MOTIF_NAME = {name!r}\n"
        f"MOTIF_DESCRIPTION = {description!r}\n"
        "def motif(fen):\n"
        "    return False\n"
    )


def fenced(source):
    return "```python\n" + source + "```"


def synthetic_positions(prefix="synthetic-board"):
    return [Position(f"{prefix}-{i}", "white", int(i < 4), str(i)) for i in range(8)]


GO_SOURCE = """MOTIF_NAME = 'More stones'
MOTIF_DESCRIPTION = 'The evaluated player has more stones.'
def motif(board_rows, color):
    other = 'W' if color == 'B' else 'B'
    return sum(row.count(color) for row in board_rows) > sum(row.count(other) for row in board_rows)
"""

CHESS_SOURCE = """import chess
MOTIF_NAME = 'More pawns'
MOTIF_DESCRIPTION = 'The evaluated player has more pawns.'
def motif(fen):
    board = chess.Board(fen)
    return len(board.pieces(chess.PAWN, board.turn)) > len(board.pieces(chess.PAWN, not board.turn))
"""


def go_row(stone, label=None):
    rows = ["." * 19 for _ in range(19)]
    rows[3] = "." * 3 + stone + "." * 15
    result = {"board_rows": rows, "color": "B"}
    if label is not None:
        result["label"] = label
    return result


def chess_row(positive, label=None):
    result = {
        "fen": (
            "7k/8/8/8/8/8/P7/K7 w - - 0 1"
            if positive
            else "7k/7p/8/8/8/8/8/K7 w - - 0 1"
        ),
        "color": "white",
    }
    if label is not None:
        result["label"] = label
    return result
