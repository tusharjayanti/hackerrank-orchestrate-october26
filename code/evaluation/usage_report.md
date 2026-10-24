# Usage report — measure run `20261024T213121Z-deletion-requests-afc3e20`

| | |
|---|---|
| Run id | `20261024T213121Z-deletion-requests-afc3e20` |
| Commit | `afc3e20` (clean tree) |
| Command | `ANTHROPIC_API_KEY= ANTHROPIC_AUTH_TOKEN= MODEL_VISION=claude-sonnet-5-5 MODEL_INTENT=claude-sonnet-5-5 MODEL_AGENT=claude-opus-5-5 .venv/bin/python code/main.py --mode measure` |
| Output | `output.csv`, 518 rows, sha256 `32c89163251480b144a68a27566ba7d1c927366a3980cf2431d6f9324ab81817` |
| Model access | `replay` (replay: cached answers only; a miss is flagged, never called) |
| Models | vision `claude-sonnet-5-5`, intent `claude-sonnet-5-5`, agent and vision tiebreak `claude-opus-5-5` |
| Budget breaker | max 200 calls / $10.00; tripped: no |
| Status mix | {'COMPLETE': 152, 'INCOMPLETE': 356, 'NOT_APPLICABLE': 10} |

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
| `claude-opus-5-5` | investigator agent | 20 | 141,680 | 3,606 | 0.6388 |
| `claude-sonnet-5-5` | intent | 15 | 50,897 | 17,033 | 0.2721 |
| `claude-sonnet-5-5` | vision (screenshots) | 22 | 92,826 | 4,924 | 0.2349 |

| Model | Calls | Input tokens | Output tokens | USD | $/MTok in / out |
|---|---|---|---|---|---|
| `claude-opus-5-5` | 20 | 141,680 | 3,606 | 0.6388 | 4.00 / 20.00 |
| `claude-sonnet-5-5` | 37 | 143,723 | 21,957 | 0.5070 | 2.00 / 10.00 |
| **Overall** | 57 | 285,403 | 25,563 | 1.1459 | |

| Per request (over all 518 requests) | Value |
|---|---|
| Model calls | 0.110 |
| Total tokens | 600.3 (input 551.0, output 49.3) |
| Cost | $0.00221 |

Total tokens across the run: 310,966. Estimated total cost of a cold run: $1.1459.

Notes: vision calls read the 11 screenshots twice each (Sonnet); intent classifies the unique request texts 25 per call (Sonnet); the agent runs only on gated cases, up to six Opus turns each. PDFs and every text format are parsed without a model.
