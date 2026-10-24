# Usage report — measure run `20261024T194721Z-deletion-requests-24896fd`

| | |
|---|---|
| Run id | `20261024T194721Z-deletion-requests-24896fd` |
| Commit | `24896fd` (clean tree) |
| Command | `ANTHROPIC_API_KEY= ANTHROPIC_AUTH_TOKEN= MODEL_VISION=claude-sonnet-5-5 MODEL_INTENT=claude-sonnet-5-5 MODEL_AGENT=claude-opus-5-5 .venv/bin/python code/main.py --mode measure` |
| Output | `output.csv`, 518 rows, sha256 `b43ff70351c803423f26efc4e357954987b41b7a81a9fb409727418a9afff7ea` |
| Model access | `replay` (replay: cached answers only; a miss is flagged, never called) |
| Models | vision `claude-sonnet-5-5`, intent `claude-sonnet-5-5`, agent and vision tiebreak `claude-opus-5-5` |
| Budget breaker | max 200 calls / $10.00; tripped: no |
| Status mix | {'COMPLETE': 154, 'INCOMPLETE': 354, 'NOT_APPLICABLE': 10} |

## 1. This run

| Model | Cache hits | Live calls | Input tokens | Output tokens | USD |
|---|---|---|---|---|---|
| `claude-opus-5-5` | 6 | 0 | 0 | 0 | 0.0000 |
| `claude-sonnet-5-5` | 37 | 0 | 0 | 0 | 0.0000 |
| **Total** | 43 | 0 | 0 | 0 | 0.0000 |

## 2. Cold cost: the live calls that produced every replayed answer

What the same run costs with an empty cache (each call priced at the list price in `ghostaudit/budget.py`).

| Model | Stage | Calls | Input tokens | Output tokens | USD |
|---|---|---|---|---|---|
| `claude-opus-5-5` | investigator agent | 19 | 125,968 | 3,140 | 0.5667 |
| `claude-sonnet-5-5` | intent | 15 | 50,897 | 17,033 | 0.2721 |
| `claude-sonnet-5-5` | vision (screenshots) | 22 | 92,826 | 4,924 | 0.2349 |

| Model | Calls | Input tokens | Output tokens | USD | $/MTok in / out |
|---|---|---|---|---|---|
| `claude-opus-5-5` | 19 | 125,968 | 3,140 | 0.5667 | 4.00 / 20.00 |
| `claude-sonnet-5-5` | 37 | 143,723 | 21,957 | 0.5070 | 2.00 / 10.00 |
| **Overall** | 56 | 269,691 | 25,097 | 1.0737 | |

| Per request (over all 518 requests) | Value |
|---|---|
| Model calls | 0.108 |
| Total tokens | 569.1 (input 520.6, output 48.4) |
| Cost | $0.00207 |

Total tokens across the run: 294,788. Estimated total cost of a cold run: $1.0737.

Notes: vision calls read the 11 screenshots twice each (Sonnet); intent classifies the unique request texts 25 per call (Sonnet); the agent runs only on gated cases, up to six Opus turns each. PDFs and every text format are parsed without a model.
