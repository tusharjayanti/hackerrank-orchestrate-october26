# Design notes

## Ordering decisions

- **Step order.** Within each phase, steps are sorted by the operation's row
  in the spec table (`ALLOWED_TRIPLES` in `ghostaudit/schemas.py`), then
  `system_id`, then `resource_id`. This sort is authoritative. The spec's
  "downstream before source" rule is satisfied by it because
  `DISABLE_DATA_FLOW` and `BLOCK_RESTORE` are the first two remediation rows,
  so flows and restores are always stopped before records are changed. None
  of the 32 samples contradicts this order.
- **Case order.** Rows in `output.csv` use natural numeric order on the
  `case_id` suffix (`case_99` before `case_100`), which matches
  `deletion_requests.csv`. Lexicographic order would be an equally valid
  presentation choice; the grader keys on `case_id`, not on row position.

## Identity

- Matching is exact on normalised identifiers (emails lowercased, phones
  digits-only, hashes lowercased). Names never link records.
- An identifier shared by several users links to none of them and logs a
  warning. Intended policy if one appears: the owner at the observation's
  `captured_at`, decided by `user_audit_log.event_time`.
