# Paper experiments

Install with `python -m pip install -e .` and run from the repository root.
The runner ships in the source distribution, not the wheel.

## Positions

Follow [data collection](data.md) and the [episode format](../docs/formats.md#evaluation-episodes-jsonl).
Each episode has four positive and four negative support positions, plus eight
held-out positions of each kind: positive, negative and nearby negative.

## BinProp

Evaluate a fixed motif library without model calls:

```sh
python -m experiments.run binprop --game chess \
  --episodes test100.jsonl --library chess/fable-5 \
  --output runs/chess-fable-binprop
```

For Go, use `--game go` and a `go/` library. For your own motifs, pass
`basis.json` as `--library`. [Discovery](../README.md#discover-motifs) uses a
separate winning/losing pool.

## Direct prompting

```sh
export NEBIUS_API_KEY='your-token-factory-key'
python -m experiments.run direct --game chess --episodes test100.jsonl \
  --model Qwen/Qwen3.5-397B-A17B \
  --base-url https://api.tokenfactory.us-central1.nebius.com/v1/ \
  --request-extra experiments/settings/tokenfactory-max.json \
  --output runs/chess-qwen-direct
```

The [prompts](prompts) include only support boards. Add `--no-tools` for
proprietary chess models; the Go prompt already includes it.

The client uses OpenAI-compatible chat completions. Native OpenAI and Anthropic
models need a compatible gateway; set `--base-url`, `--api-key-env` and
`--request-extra` accordingly. Token Factory settings are not portable.

## Results and resuming

Each run saves settings, responses, selected functions and episode scores.
Repeat the same command with `--resume` to continue. Inputs and settings must
match; finished episodes are skipped. New runs require new output directories.

See [selection, scoring, retries and execution limits](../docs/method.md).
Run generated code in an isolated container without secrets or network access.

## Model settings

The paper used these model IDs and reasoning settings:

| Model | Model ID | Settings |
| --- | --- | --- |
| Qwen | `Qwen/Qwen3.5-397B-A17B` | `tokenfactory-max.json`; US Central |
| GLM | `zai-org/GLM-5.2` | `tokenfactory-max.json`; UK South |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro` | `tokenfactory-max.json`; UK South |
| Kimi | `moonshotai/Kimi-K3` | `kimi-max.json`; global endpoint; 900s timeout |
| GPT-5.5 | `gpt-5.5` | Responses API: `xhigh` effort |
| GPT-5.6 Sol | `gpt-5.6-sol` | Responses API: `xhigh` effort |
| Opus 5 | `claude-opus-5` | Adaptive thinking: `max` effort |
| Fable 5 | `claude-fable-5` | Adaptive thinking: `max` effort |

Reasoning and answer budgets were 61,440 and 4,096 tokens. This runner requests
65,536 total tokens; it cannot enforce separate limits on combined-budget
endpoints. Temperature and top-p are 1 where supported.
