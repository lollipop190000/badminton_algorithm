# Budget CQL implementation ledger

Approved spec: conversation plan, narrowed by user on 2026-11-04: teammate owns 09/10 integration; implement CQL and its handoff first.

- Branch: research/03-budget-cql-comparison, from origin/main. Existing independent checkout is used in place to obey the workspace Git boundary; original research branch and artifacts are preserved.
- Ruling: final integrated costs are an external dependency. A provisional existing 10/M0 cost bundle is permitted only by explicit CLI flag for connection tests; it is never labeled an integrated/final comparison.
- Ruling: reset budgets at the first eligible middle-rally state, avoiding unmodeled opening/service costs. Quantiles use complete eligible suffixes from training games only.
- Ruling: terminal Z0 has no recorded intended zone. Preserve terminal losses by fractional target completion from the train-only conditional continuing-zone distribution, restricted to the common mask. Label these rows explicitly; this relies on the same zone-independent termination assumption as the mathematical strategy. No outcome/next-shot information enters the state.
- Tasks: common contract and replay; training and frozen policy adapter; diagnostic evaluation/FQE; reference-cost connection run; reports and repository checks; independent final review.
- RED: new contract tests written before production implementation.
- Task 1/2/3: common contract, replay, training and frozen-policy auxiliary FQE implemented. Provisional v1/v2 development runs remain in ignored runtime; they are not final integrated results and will not be published.
- Final review: independent reviewer found three Important defects: mixed-order terminal priors, unchecked training artifacts at evaluation, and scoped sync erasing unrelated omission metadata. All three have dedicated failing regression tests before the fixes.
- Final: fixed prior by using one common history order per history/shot and saving its order; fixed frozen training/dataset hash validation; fixed omission metadata merge in scoped sync.
- Final: minor (deferred): set_id is encoded in existing globally unique rally_id rather than added as another explicit replay/merge key. Current audited data is unaffected; a new importer that reuses rally IDs across sets must extend the key before use.
- Ruling: commit tested sources before the publishable connection run so RUN.json identifies the executed source commit. Do not rewrite development runs as if they used that later commit.
