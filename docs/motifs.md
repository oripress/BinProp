# Chess motif libraries

The paper's eight libraries contain **192 motifs**, with their original names,
descriptions, functions, and discovery scores.

| Library ID | Model | Motifs |
| --- | --- | ---: |
| `chess/claude-opus-5` | Claude Opus 5 | 30 |
| `chess/deepseek-v4-pro` | DeepSeek-V4-Pro | 28 |
| `chess/fable-5` | Fable 5 | 20 |
| `chess/glm-5.2` | GLM-5.2 | 21 |
| `chess/gpt-5.5` | GPT-5.5 | 44 |
| `chess/gpt-5.6-sol` | GPT-5.6 Sol | 21 |
| `chess/kimi-k3` | Kimi K3 | 11 |
| `chess/qwen3.5-397b-a17b` | Qwen3.5-397B-A17B | 17 |

Pass a library ID or JSON file to `infer --library`. See the
[quickstart](../README.md#explain-your-positions) for an example and
[input formats](formats.md#library-json) for Python access and export.

## Scores and provenance

Motifs are ordered by [discovery delta](method.md#discovery), not held-out explainability.

All eight libraries come from 100-iteration runs on the same discovery positions
with the `max_64k_total` reasoning profile. The Kimi library is from the completed
September 21, 2026 run. See [model settings](../experiments/README.md#model-settings)
and [data collection](../experiments/data.md#chess).

Tests validate the libraries and their functions, but do not rerun the paper's
engine evaluations.
