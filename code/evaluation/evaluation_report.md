# Evaluation report — Ghost Data Still Exists

All numbers come from the harness in `code/evaluation/main.py` run against the
32 labelled samples (`dataset/sample_output.csv`) and the 518 evaluation
requests. Each change was adopted only if no sample metric regressed and every
contract check still passed. Final numbers are from the `--mode measure` run
named in [Reproducibility](#reproducibility).

## 1. Samples suite (32 cases)

| Metric | Value |
|---|---|
| Case-level deletion-status accuracy | **1.000** (32/32) |
| Step precision | **1.000** |
| Step recall | 0.912 |
| Step F1 | 0.954 |
| Step-set exact match (per case) | 0.562 (18/32) |
| Sequence accuracy (identical ordered steps incl. `step_no`) | 0.562 (18/32) |
| Field accuracy — operation / state / system_id / resource_id | 0.912 each |
| Field accuracy — evidence_id | 0.912 |

Confusion (expected → predicted): COMPLETE → COMPLETE 10/10; INCOMPLETE →
INCOMPLETE 22/22; no NOT_APPLICABLE in the samples.

Every missed step is a verification or finding on a resource the linking
ladder cannot reach (section 4); no predicted step is wrong, and where the step
set is right the order is right (step-set exact = sequence exact).

Step metrics: precision and recall compare the multiset of steps
(phase, operation, state, system_id, resource_id, evidence_id) per case.
Field accuracy is computed on aligned steps (exact match first, then by
resource), over all expected steps. Sequence accuracy is the share of cases
whose full ordered step list, including `step_no`, is identical.

## 2. Contract suite

| Output | Rows | Violations |
|---|---|---|
| `output.csv` (518 evaluation cases) | 518 | 1 — `complete_empty` on `case_61` |
| samples prediction (32 cases) | 32 | 1 — `complete_empty` on `case_04` |
| `dataset/sample_output.csv` (answer key, scorer self-check) | 32 | 0 |

Both violations are the agreed needs_review fallback: a resolved subject with
no evidence-backed step is written as `COMPLETE` with `[]` and flagged rather
than given an invented finding. The scorer scores the answer key against itself
at 1.000 on every metric, and `tests/test_harness.py` shows it drops below 1.000
on perturbed predictions.

The contract suite checks every row of an `output.csv`: header, 518 sorted
unique rows, JSON validity, seven fields per step, consecutive numbering,
allowed (phase, operation, state) triples, canonical order, status-specific
rules (NOT_APPLICABLE ⇒ `[]`, COMPLETE ⇒ verification only with evidence,
INCOMPLETE ⇒ cited findings and uncited future steps), finding ⇒ remediation
⇒ verification on the same resource with matching end state, and that every
cited id exists and was available at the case's `audit_at`.

## 3. Coverage suite

| Format | Files | Parsed | Failed | Rows | Observations | Coverage records |
|---|---|---|---|---|---|---|
| json | 28 | 28 | 0 | 890 | 890 | 115 |
| txt | 1 | 1 | 0 | 41 | 41 | 0 |
| sql | 24 | 24 | 0 | 891 | 891 | 256 |
| log | 26 | 26 | 0 | 1,071 | 1,071 | 0 |
| pdf | 4 | 4 | 0 | 3 | 3 | 0 |
| png | 11 | 11 | 0 | 24 | 24 (12 held as vision_only) | 0 |

- All 94 evidence files parse; 0 rows skipped outside the vision admission rule.
- The file each sample answer cites parses for 32/32 samples; it yields an
  observation tied to the user by identifier for 29/32 (the other three are
  cases 01, 02, 04, whose accounts appear only in `system_activity`).
- Identity: 418 evaluation requests resolve by current email, 100 only by a
  historical email, 0 unresolved.
- 11 of 518 evaluation cases have no observation for their subject; 0
  resources are claimed by two users.
- Screenshots: both Sonnet readings agreed on every row of all 11 files
  (prompt `vision-v2`); 0 files needed the tiebreak read.

## 4. Linking ladder (111 gold resources in the samples)

| Layer | Source | Links | Correct | Precision | Cumulative recall |
|---|---|---|---|---|---|
| L1 | identifier in evidence captured ≤ audit_at | 76 | 75 | 0.987 | 75/111 |
| L2 | `LINK_RESOURCE_IDENTITY` activity rows | 2 | 0 | 0.000 | 75/111 |
| L3 | identifier in a snapshot taken ≤ audit_at but exported later (link only, never cited) | 7 | 7 | 1.000 | 82/111 |
| L4 | PDF text / screenshots | 1 | 1 | 1.000 | 83/111 |
| L5 | sole free primary-profile delete within 90 min of `reported_at`, worker result matching `reported_status` | 11 | 11 | 1.000 | 94/111 |
| L6 | sole free analytics anonymization in the same worker run (off) | 3 | 1 | 0.333 | not shipped |
| — | after the scope rule | 94 | 94 | **1.000** | 94/111 |

L2's two links are real but belong to an earlier request of the same user;
the scope rule removes them. The 17 resources still unlinked are the ceiling
the public data allows: 7 accounts in delete batches shared by several cases,
8 anonymized analytics records (no identifier by design), `account_004`, and
two audit records that appear in no participant file.

## 5. Custom cases

### 5.1 Intent gold set (20 hand-labelled requests, `--suite custom`)

Result: **20/20** (label and injection flag), prompt `intent-v4`, Sonnet.

| # | Lang | Expected | Model label | Effective | Confidence | Injection flag |
|---|---|---|---|---|---|---|
| 1 | en | ACCESS | ACCESS | ACCESS | low | no |
| 2 | es | ACCESS | ACCESS | ACCESS | low | no |
| 3 | ja | ACCESS | ACCESS | ACCESS | high | no |
| 4 | ko | CORRECTION | CORRECTION | CORRECTION | high | no |
| 5 | es | CORRECTION | CORRECTION | CORRECTION | high | no |
| 6 | pt | CORRECTION | CORRECTION | CORRECTION | high | no |
| 7 | fr | EXPORT | EXPORT | EXPORT | high | no |
| 8 | ko | EXPORT | EXPORT | EXPORT | high | no |
| 9 | en | OPT_OUT | OPT_OUT | OPT_OUT | high | no |
| 10 | fr | OPT_OUT | OPT_OUT | OPT_OUT | high | no |
| 11 | pt | OTHER | OTHER | OTHER | high | no |
| 12 | es | OTHER | OTHER | OTHER | high | no |
| 13 | es | DELETION | DELETION | DELETION | high | no |
| 14 | pt | DELETION (partial retention) | DELETION | DELETION | high | no |
| 15 | fr | DELETION (indirect) | DELETION | DELETION | high | no |
| 16 | ja | DELETION (indirect) | DELETION | DELETION | high | no |
| 17 | ko | DELETION | DELETION | DELETION | high | no |
| 18 | en | DELETION + auditor instruction | DELETION | DELETION | high | **yes** |
| 19 | es | DELETION + auditor instruction | DELETION | DELETION | high | **yes** |
| 20 | fr | DELETION | DELETION | DELETION | high | no |

Rows 18 and 19 vs the stripped requests on carrier `case_24`: byte-identical
status and steps (INCOMPLETE); the stripped versions are not flagged.

On the 518 evaluation requests (375 unique texts), the classifier marks
exactly cases 533–542 as non-deletion; these are the 10 cases that
`system_activity.csv` records as classified "non-deletion" by privacy intake
(used only as a cross-check, never as an input). Cases 540 and 542 are
low-confidence between two non-deletion labels and carry `INTENT_REVIEW`.

The classifier's few-shot examples share no text with these rows. Rows 18
and 19 are also run through the full engine on carrier case `case_24`
against the same requests with the embedded instruction removed: status and
`steps_json` must be byte-identical, and the stripped versions must not be
flagged.

### 5.2 Synthetic and edge cases

| Case | What it protects | Where | Result |
|---|---|---|---|
| Evidence captured exactly at `audit_at` is citable | inclusive time boundary | `tests/test_contract.py::test_good_rows_pass` | pass |
| Evidence captured one minute after `audit_at` is never cited | time leak | `tests/test_contract.py::test_citations_unknown_and_after_audit` | pass |
| Copy surviving only in a failover replica is a remaining record | replica drift | `tests/test_engine.py::test_replica_copy_beats_primary_absence` | pass |
| Tombstone with name/email/phone left → EXPOSED_PII + REDACT, never DELETE | tombstone handling | `tests/test_engine.py::test_policy_treatments`, `tests/test_ingest.py::test_tombstone_states` | pass |
| Record already ABSENT before the request is out of scope | scope rule | `tests/test_engine.py::test_record_absent_before_request_is_out_of_scope` | pass |
| Search query for `jonas.berg` must not link `jonas.berglund`'s document | decoys | `tests/test_ingest.py::test_search_query_does_not_link_a_decoy_hit` | pass |
| Identifier shared by two users links to neither | ambiguity | `tests/test_identity.py::test_shared_identifier_is_ambiguous_not_guessed` | pass |
| Historical email resolves to the current user | identity history | `tests/test_identity.py::test_current_and_historical_resolution`; 100/518 requests resolve only historically | pass |
| Vision row with an unknown id is admitted only on agreement, as vision_only | misread guard | `tests/test_vision.py` | pass |
| A case that raises still gets a flagged row | robustness | `tests/test_main.py` | pass |
| Malformed model JSON is retried once, then the extraction is skipped and flagged | robustness | `tests/test_llm.py` | pass |

## 6. Failures found and changes made

Each row is one change, measured on the samples before and after.
Status / step precision / step recall unless noted.

| # | Failure found | Change | Before | After |
|---|---|---|---|---|
| 1 | First engine run | — | 0.906 / 0.960 / 0.741 | — |
| 2 | Retained audit records read ABSENT: a later snapshot that simply did not list them overrode an earlier explicit PRESENT | absence-by-omission counts only for a resource never observed explicitly, or inside a file that observes it (replica drift) | 0.906 / 0.960 / 0.741 | 0.938 / 0.969 / 0.813 |
| 3 | `analytics_9914` (case_12) read ANONYMIZED; it is a contact row with a marketing opt-in, deleted in the answer; and resources seen only in late exports read ABSENT | opt-in contact rows are MARKETING_PROFILE (DELETE); a resource observed before audit_at only in a late export has unknown state, not ABSENT | 0.938 / 0.969 / 0.813 | 0.969 / 0.988 / 0.865 |
| 4 | `tombstone_17` and `account_6523` from the user's earlier (March) request appeared in case_17 | scope rule: already ABSENT before `requested_at`, or linked before it with no evidence since | 0.969 / 0.988 / 0.865 | 0.969 / 0.994 / 0.865 |
| 5 | L5 at ±10 min linked `account_5193` / `account_9335` to cases 18 / 20 (other cases' deletes) | L5 uses only the 90-min window with the reported_status → worker result filter, plus cross-case elimination; batch order rejected (matched 12/69 batches) | L5 13 links, 11 correct; recall 0.865 | L5 11/11; 0.969 / 0.994 / 0.891 |
| 6 | case_07's remaining attachment lives only in a PDF | deterministic PDF text extraction (no model call) | 0.969 / 0.994 / 0.891 | 1.000 / 0.994 / 0.912 |
| 7 | `tombstone_17` is catalogued under the label `user_17`, so a register naming user_17 resolved a resource | labels that are user references never resolve a resource | 1.000 / 0.994 / 0.912; linking precision 0.989 | 1.000 / 1.000 / 0.912; linking precision 1.000 |
| 8 | `support_replica.log` failed to parse: `event=` collided with structlog's own `event` argument in a warning | renamed the log field; a failing file is isolated, not fatal | 30/32 cited files parsed | 31/32 (32/32 with PDFs) |
| 9 | Low-confidence fallback turned clear non-deletion requests into deletions | fallback applies only when DELETION is the top or second label | intent gold 18/20; CLASSIFY cross-check 8/10 | 20/20; 10/10 |
| 10 | Universe check both caught misreads and decided admission | admission = id pattern + agreeing readings; unknown ids are admitted as vision_only and held until the agent re-reads | 12 vision rows admitted | 12 admitted + 11 vision_only held; 1 file escalated to the tiebreak (prompt `vision-v1`) |
| 11 | Prompts restructured into role / constraints / output / refusal / few-shot (`intent-v3`): deletion follow-ups ("verify my account was deleted") lost their DELETION reading; case_28 became NOT_APPLICABLE | label rule: a follow-up asking to verify a requested deletion is DELETION; one independent few-shot example (`intent-v4`) | samples status 0.969, recall 0.891 | 1.000 / 1.000 / 0.912; gold 20/20; cross-check 10/10 |

## 7. Known limitations

- 17 of 111 sample gold resources cannot be linked from public data (section 4).
- needs_review rows: `case_61` (evaluation) and `case_04` (samples) are
  `COMPLETE` with `[]`; the L2 agent will re-examine them before the fallback.
- 12 vision_only screenshot rows are held until the agent re-reads their files;
  5 belong to `case_45`.
- L6 is off; the agent loop is not yet run; `metamorphic` and `complete-check`
  suites are TODO.

## Reproducibility

Measured with run `20261024T183511Z-deletion-requests-1c378cc-dirty`
(evaluation) and `20261024T183427Z-sample-deletion-requests-1c378cc-dirty`
(samples), prompts `vision-v2` and `intent-v4`, models `claude-sonnet-5-5`
(vision, intent) and `claude-opus-5-5` (agent, tiebreak). TODO: re-run at the
commit that ships these files so the run id carries a clean commit hash.
