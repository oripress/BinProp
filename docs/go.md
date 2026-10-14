# Go motif libraries

The paper's eight Go libraries contain **250 motifs**:

| Library ID | Motifs |
| --- | ---: |
| `go/claude-opus-5` | 35 |
| `go/deepseek-v4-pro` | 5 |
| `go/fable-5` | 45 |
| `go/glm-5.2` | 34 |
| `go/gpt-5.5` | 39 |
| `go/gpt-5.6-sol` | 31 |
| `go/kimi-k3` | 30 |
| `go/qwen3.5-397b-a17b` | 31 |

## Offline example

After [installation](../README.md#quickstart), run:

```sh
python examples/explain_go.py
```

The example replays a JGDB game and two saved KataGo continuations, then selects
Opus 5 motifs. It needs no engine or model call. To try another library:

```sh
python examples/explain_go.py --write-comparison go-comparison.json
binary-properties infer --library go/kimi-k3 --positions go-comparison.json
```

## Position format

Use `board_rows` and `color` as described in [input formats](formats.md).
Keep the evaluated color fixed throughout a comparison, regardless of who moves next.

The paper uses row 0 as GTP rank 19 and columns A-T, skipping I. Boards are
reflected vertically relative to conventional SGF display; the example uses
this orientation for both boards and continuation coordinates.

Some motifs have fixed komi assumptions. Kimi's `local stone density territory`
uses a 7.5-point adjustment; Opus's `careful recount of a board that looks level`
uses 6.5-point thresholds. The interface has no komi argument, so these remain
unchanged even when the engine uses a different komi. Go validation checks
Boolean outputs for both colors, but does not require color-swap invariance.

## Example provenance

The SGF is embedded in [the example](../examples/explain_go.py):
`jgdb/sgf/test/0002/00002616.sgf` from `jgdb_public_domain_v1`.
It records Black 7d, White 8d, komi 0.5, and W+4.50, but no names or date.
SHA256: `f52d378492b497abadb421ce1b484ce46a49613866d182ce7a632681ccd03c4d`.

Episode `go_test_00002616_move149` starts after 149 moves, with White to play.
Its saved support continuations are:

- Positive: `N7 T7 S6 T5 T4 Q7 R5 S4`, evaluation +8.7209 points.
- Negative: `P3 Q4 N7 O6 M6 P6 P5 P7`, evaluation -1.6971 points.

These are saved KataGo variations, not moves played in the game. Evaluations
use White's perspective, Chinese rules, and komi 0.5; the SGF does not specify rules.
