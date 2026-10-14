# CuongDanh verification — 2026-10-14

Upstream checked through GitHub API: VinUni-AI20k/Day26-Colosseum-Agent-Arena-Kit
and local HEAD both equal 6a68efdd25a172f3c6b2f62fc3627f3215b06984.
No merge, push or protected-file edit was performed.

## Fixed and regression-tested

- Low-credit wildcard crash and omitted row costs; grammar row limits are integers.
- A2A signed JSON token parsing, signature/replay checks and peer/card/skill binding.
- Provenance from history, correct Learner write resource, stale-header rejection,
  stable logical idempotency keys, and retries only after failed outcomes.
- Drift no longer implies canonical freshness; explicit comparison anchors work.
- Real answer gate in the opt-in agent.runtime.Agent, before environment recording.
- Source-aware private-text redaction, required fields, decimal arithmetic and
  numeric precision checks. Unrelated poisoned material does not veto clean answers.
- Prosecutor evaluates executed calls, not superseded pre-rewrite commands.
  Read-only progress, valid leases/writes, selected search results, replica
  comparisons and corrected precondition retries are covered by negative tests.
- atk_02 now asks day22 DPO loss and mutates the matching day22 DPO/ORPO deck.

## Checks

- Team regressions: 32 passed, standard-library unittest.
- Public suite: 4608 passed, 7 skipped, 1 failed (upstream Windows libc.open).
- Deck: 0 FAIL, 1 explicit lethality WARN.
- Prosecutor labelled fixtures: precision 1.000, recall 0.471, F1 0.640,
  false-claim rate 0.000. Stale-read evidence remains intentionally conservative;
  eight semantic hooks remain unimplemented, not silently counted as covered.
- Spar smoke: rookie/all 100–0, adversary/prosecutor 11–0, seed 1.
  These results do not verify answer-time enforcement or real attack lethality.
- No-key gate passes. Protected kit/, bots/, fixtures/ remain unchanged.

## Remaining host/platform dependencies

The supplied spar.py fabricates its final answer and never loads the team Agent.
The tournament host must explicitly use agent.runtime.Agent (or equivalent
answer hook). The wrapper is tested end to end locally; merely bundling it does
not establish that the instructor host will select it.

The protected isolation baseline is not Windows-compatible. Run the complete
isolation suite on the instructor's supported OS; do not modify or skip kit
checks and call that a full pass.

On Windows the venv path is .venv/Scripts. Make needs BIN=.venv/Scripts and a
working POSIX shell for validate/submit. Equivalent packaging entry point:

    .\.venv\Scripts\python.exe -m kit.submit --team CuongDanh

The updated bundle is a local build, not an external tournament submission.
