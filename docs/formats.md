# Input and output formats

## Discovery input: JSONL

Each line is a position with a winning or losing `label`; `id` is optional.
Labels can be `1` / `"winning"` or `0` / `"losing"`.
The examples below illustrate the format; use engine evaluations for actual labels.

### Chess

```json
{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","color":"white","label":1}
```

`color` is `white` or `black`, defaulting to the FEN's side to move.
The predicate receives a FEN with its turn set to that color.
Keep the evaluated player fixed across continuation positions.

### Go

`board_rows` contains 19 strings of 19 characters: `.` for empty, `B` for Black,
and `W` for White. `color` must be `B` or `W`; the label is for that player.
Rows retain their supplied orientation and colors.

```python
position = {
    "board_rows": ["." * 19 for _ in range(19)],
    "color": "B",
    "label": 1,
}
```

Motifs receive neither move history, komi, nor ko state; these can still matter
when obtaining engine labels.

## Inference input: JSON

```json
{
  "positive": [{"fen": "YOUR_POSITIVE_FEN", "color": "white"}],
  "negative": [{"fen": "YOUR_NEGATIVE_FEN", "color": "white"}]
}
```

Both lists must be nonempty. For Go, use `board_rows` and `color` instead of FENs.
Labels are ignored. Keep held-out positions separate if evaluating generalization.

## Evaluation episodes: JSONL

The experiment runner accepts JSONL or gzipped JSONL, with one episode per line:

```json
{
  "id": "game-and-move",
  "support": {"positive": [], "negative": []},
  "held_out": {"positive": [], "negative": [], "nearby_negative": []}
}
```

Fill each support list with four positions and each held-out list with eight.
Use the chess or Go position objects above. IDs must be unique. Keep the evaluated
player fixed; chess positions must have that player to move. All boards must be
distinct, with no overlap between groups.

The loader also accepts the paper's continuation-tree fields: `root_turn`
(chess) or `root_to_play` (Go), `positive_support_pool`, `negative_support_pool`,
`positive_holdout`, `normal_negative_holdout`, and `counterfactual_negative_holdout`.
It selects four evenly spaced positions from each support pool and removes them
from the held-out groups, which must then contain eight positions each.

## Predicate contract

Chess:

```python
import chess

MOTIF_NAME = "Material advantage"
MOTIF_DESCRIPTION = "The evaluated player has more non-king material."

def motif(fen):
    board = chess.Board(fen)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9}
    ours = sum(v * len(board.pieces(piece, board.turn)) for piece, v in values.items())
    theirs = sum(v * len(board.pieces(piece, not board.turn)) for piece, v in values.items())
    return ours > theirs
```

Go:

```python
MOTIF_NAME = "More stones"
MOTIF_DESCRIPTION = "The evaluated player has more stones on the board."

def motif(board_rows, color):
    opponent = "W" if color == "B" else "B"
    return sum(row.count(color) for row in board_rows) > sum(
        row.count(opponent) for row in board_rows
    )
```

These examples illustrate the interfaces, not the paper's motifs. Functions must
return a Boolean. Names are 1-180 characters, descriptions 1-2,000, and source
1-12,000. Chess functions must pass color-mirror checks. Go functions need not
be color-swap invariant; see [komi assumptions](go.md#position-format).

## Library: JSON

`basis.json` contains `schema_version: 1`, `game`, `motifs`, and `metadata`.
Each motif stores its `name`, `description`, Python `source`, discovery `delta`,
and source-derived `id`. No position data is embedded. A library file is enough
for offline inference; new discovery runs recalculate deltas on their own data.

Use a library ID with `infer --library` or `discover --initial`, or load it in Python:

```python
from binary_properties.libraries import load_builtin_library

library = load_builtin_library("chess/kimi-k3")
```

Export with `binary_properties.io.save_library(path, library)`. The
[chess](motifs.md) and [Go](go.md) catalogs give counts and sources.

## Inference output

`selected` contains the chosen motifs, `ranked` all successfully evaluated
eligible motifs, and `errors` any execution failures. Each motif includes its
metadata, source, delta, positive/negative rates, support fit, and weighted score.

`support_counts` gives group sizes; `qualifying_count` counts motifs meeting the
fit threshold. `used_fallback` means none qualified and the best eligible motif
was returned anyway. Motifs with nonpositive delta are excluded. If every
eligible motif fails to run, the CLI writes the errors and exits with status 2.
Existing output files are not overwritten.
