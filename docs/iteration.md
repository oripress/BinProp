# Discover and extend motif libraries

Use a [winning/losing pool](formats.md#discovery-input-jsonl) separate from evaluation positions.

## Start a new run

After [installation](../README.md#quickstart), choose a model on
[Nebius Token Factory](https://tokenfactory.nebius.com/):

```sh
export NEBIUS_API_KEY='your-token-factory-key'
export MOTIF_MODEL='your-token-factory-model-id'

binary-properties discover \
  --game chess \
  --positions discovery.jsonl \
  --initial chess/kimi-k3 \
  --output runs/kimi-next \
  --iterations 20
```

This starts with the paper's Kimi motifs and recomputes their deltas on your data.
Omit `--initial` to start from scratch, or pass an earlier `basis.json`.
The original library is unchanged.

Accepted motifs are saved in `basis.json`; decisions in `iterations.jsonl`.
Use the result for [inference](../README.md#explain-your-positions) with separate comparison positions.

## Resume the same run

Finish an interrupted run, or add 20 iterations:

```sh
binary-properties discover --resume runs/kimi-next
binary-properties discover --resume runs/kimi-next --iterations 20
```

Resume restores settings and feedback from `checkpoint.json` and skips completed
iterations. Export the API key again. The input must be unchanged; if it moved,
pass `--positions /new/path.jsonl`.

For different data, models or settings, start a new run with
`--initial runs/kimi-next/basis.json`. Keep checkpoints private: they contain
feedback boards and request options, but not API keys.

## Provider options

The default endpoint is `https://api.tokenfactory.nebius.com/v1/`.
For another OpenAI-compatible provider, set `--base-url` or `MOTIF_BASE_URL`,
and `--api-key-env YOUR_KEY_VARIABLE` if needed. Keys come from the environment;
`.env` files are not loaded.

`--request-extra options.json` adds provider options, such as
[temperature](../examples/request-options.json). See
[paper model settings](../experiments/README.md#model-settings) for reasoning budgets.
Discovery thresholds and acceptance rules are in [Method](method.md#discovery).
