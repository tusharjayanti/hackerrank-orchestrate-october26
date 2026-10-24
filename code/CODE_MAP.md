# Code map

One line per module: its job, then the functions that matter. Paths are under
`code/`. A case flows top to bottom: `main.py` → `engine/decide.py` →
(linking, state, policy, steps) → optional `agent/` → `guards.py` → `output.csv`.

## Entry points

| Module | Job | Key functions |
|---|---|---|
| `main.py` | Runs one request file end to end: intent, decide, guard, write rows, traces and run manifest | `decide_all`, `engine_default`, `main` |
| `evaluation/main.py` | The six suites (contract, samples, coverage, custom, metamorphic, complete-check) | `suite_contract`, `suite_samples`, `suite_coverage` |
| `evaluation/usage.py` | Writes `usage_report.md` for a run from the call log (replay and cold cost) | `originating_calls`, `main` |
| `package.sh` | Builds `code.zip` from tracked files with `git archive`, refusing unsafe contents | — |

## Shared model

| Module | Job | Key functions |
|---|---|---|
| `ghostaudit/schemas.py` | Every enum and model; `ALLOWED_TRIPLES` is the single source of step order and validity; the `Flag` enum | `step_sort_key`, `number_steps`, `serialize_steps`, `FlagSet` |
| `ghostaudit/identity.py` | Current and historical identifiers per user, exact-match reverse map, ambiguity | `IdentityIndex.build`, `IdentityIndex.owner_of`, `normalize_identifier` |
| `ghostaudit/obs.py` | structlog setup | `get_logger` |

## L0 Ingestion (`ghostaudit/ingest/`)

| Module | Job | Key functions |
|---|---|---|
| `manifest.py` | Evidence manifest and the citation registry (what may be cited, from when) | `Manifest.load`, `EvidenceFile.eligible`, `CitationRegistry.citable_at` |
| `observations.py` | The one classifier every parser feeds (system, category, state, identifiers) and the observation table | `observe_record`, `observe_ledger`, `build_table`, `ObservationTable.case_observations` |
| `parse_records.py` | JSON shapes (service exports, snapshot bundles, collections, ledgers) and the handover TXT | `parse` |
| `parse_sql.py` | SQLite snapshots: primary/replica tables, coverage for absence by omission | `parse` |
| `parse_logs.py` | Seven log syntaxes to key/value; started/completed lines joined by trace id | `tokenize`, `parse` |
| `parse_vision.py` | PDFs by text; PNGs by two agreeing model readings with a tiebreak; vision_only admission | `parse`, `Universe` |
| `vision_prompts.py` | Screenshot transcription prompt (versioned constants) | `SYSTEM`, `SCHEMA` |
| `activity.py` | `system_activity.csv`: catalog links, intake classifications, deletion-worker events | `Activity.load`, `Activity.primary_profile_deletes` |

## L1 Audit engine

| Module | Job | Key functions |
|---|---|---|
| `ghostaudit/linking.py` | The linking ladder L1–L6, with L5/L6 solved jointly across all requests | `Linker.links`, `Linker.prepare`, `Linker.fact_links` |
| `ghostaudit/engine/state.py` | A resource's state at `audit_at` from facts captured by then; scope rule | `state_at`, `facts_for`, `out_of_scope` |
| `ghostaudit/engine/policy.py` | `deletion_rules.csv` applied to states: finding, remediation, verification | `treat`, `policy_diff`, `retention_expired` |
| `ghostaudit/engine/steps.py` | Treatments to ordered, numbered steps and the status | `build_steps` |
| `ghostaudit/engine/decide.py` | Per case: intent gate → identity → links → scope → state → policy → steps → agent call site | `World.load`, `decide_case` |
| `ghostaudit/intent.py` | Request intent (enum), batched, with the DELETION fallback only on DELETION hesitation | `classify` |
| `ghostaudit/intent_prompts.py` | Intent prompt (versioned constants) | `SYSTEM`, `FEW_SHOT`, `SCHEMA` |

## L2 Investigator agent (`ghostaudit/agent/`)

| Module | Job | Key functions |
|---|---|---|
| `candidates.py` | The gate (three conditions), engine-generated candidates, the monotone admissibility rule | `needs_agent`, `generate_candidates`, `inadmissible` |
| `loop.py` | Six-turn tool loop on the agent model; fallbacks; trace and cache | `review_case`, `run_agent` |
| `tools.py` | Five strict tools; `apply_policy` grounds a missed record, `submit_decision` accepts admissible candidates only | `run_tool`, `apply_policy`, `submit_decision` |
| `prompts.py` | Agent prompt (versioned constants, two worked examples) | `SYSTEM`, `case_message` |

## L3 Guardrails and L4 observability

| Module | Job | Key functions |
|---|---|---|
| `ghostaudit/guards.py` | Contract, citation guard, COMPLETE re-derivation and downgrade, regenerate/repair, cross-row | `guard_decision`, `check_row`, `complete_check` |
| `ghostaudit/llm.py` | The only model client: schema-constrained JSON, content-keyed cache, retries, call log, live/replay/none | `LLM.json_call`, `model_for`, `model_access` |
| `ghostaudit/budget.py` | Call and spend breaker | `Budget.check`, `Budget.record` |
